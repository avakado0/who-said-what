"""Offline interview transcription: pyannote diarization + faster-whisper ASR."""
import argparse, gc, json, os, subprocess, sys, tempfile, time
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXTRA = {}  # extra faster-whisper kwargs set from CLI flags
DIAR_REPO = "pyannote/speaker-diarization-community-1"


def to_wav(src: Path, dst: Path) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                    "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst)], check=True)


def diarize(wav: Path, n_speakers):
    import torch
    from scipy.io import wavfile
    from pyannote.audio import Pipeline
    sr, data = wavfile.read(wav)
    waveform = torch.from_numpy(data.astype("float32") / 32768.0).unsqueeze(0)
    pipe = Pipeline.from_pretrained(DIAR_REPO, token=os.environ.get("HF_TOKEN"))
    pipe.to(torch.device("cuda"))
    kw = {"num_speakers": n_speakers} if n_speakers else {}
    out = pipe({"waveform": waveform, "sample_rate": sr}, **kw)
    ann = getattr(out, "speaker_diarization", out)
    turns = [(t.start, t.end, spk) for t, _, spk in ann.itertracks(yield_label=True)]
    del pipe, out
    gc.collect(); torch.cuda.empty_cache()
    return turns


def diarize_cached(wav: Path, n_speakers, cache: Path, redo=False):
    if cache.exists() and not redo:
        return [tuple(t) for t in json.loads(cache.read_text())]
    cache.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-I", str(Path(__file__).resolve()), "--diarize-worker",
                    str(wav), str(cache), str(n_speakers or 0)], check=True)
    return [tuple(t) for t in json.loads(cache.read_text())]


def transcribe(wav, model_name, device, lang, glossary, beam0=5, compute=None):
    for beam in [b for b in (5, 2, 1) if b <= beam0]:
        try:
            return _transcribe(wav, model_name, device, lang, glossary, beam, compute)
        except RuntimeError as e:
            if "out of memory" not in str(e) or beam == 1:
                raise
            print(f"      OOM at beam {beam}, retrying with smaller beam", flush=True)
            gc.collect()


MIN_LANG_SECONDS = 3.0   # shorter segments inherit previous language (detector unreliable)
MIN_LANG_CONF = 0.85     # less confident detections inherit previous language


def detect_seg_lang(model, audio, start, end, prev, sr=16000, pad=0.3):
    """Compare only Turkish vs English probabilities for this segment's audio.
    Short or low-confidence chunks inherit the previous segment's language."""
    if end - start < MIN_LANG_SECONDS:
        return prev, 0.0
    a = audio[max(0, int((start - pad) * sr)):int((end + pad) * sr)]
    if len(a) < sr:
        return prev, 0.0
    _, _, probs = model.detect_language(audio=a)
    d = dict(probs)
    tr, en = d.get("tr", 0.0), d.get("en", 0.0)
    tot = tr + en
    if tot < 0.2:                      # neither looks like tr/en: keep previous
        return prev, 0.0
    best, conf = ("tr", tr / tot) if tr >= en else ("en", en / tot)
    return (best, conf) if conf >= MIN_LANG_CONF else (prev, conf)


def _transcribe(wav, model_name, device, lang, glossary, beam, compute=None):
    from faster_whisper import WhisperModel
    ctype = compute or ("int8_float16" if device == "cuda" else "int8")
    model = WhisperModel(model_name, device=device, compute_type=ctype)
    from scipy.io import wavfile  # pass array: avoids av/decode_audio incompatibility
    _, pcm = wavfile.read(wav)
    audio = pcm.astype("float32") / 32768.0
    segs, info = model.transcribe(
        audio, language=lang, multilingual=lang is None, beam_size=beam,
        word_timestamps=True, vad_filter=True, condition_on_previous_text=False,
        hotwords=glossary or None, **EXTRA)
    words = []
    for s in segs:
        if lang:                       # forced: tag = forced language
            seg_lang, conf = lang, 1.0
        else:
            seg_lang, conf = detect_seg_lang(model, audio, s.start, s.end, words[-1]["lang"] if words else "tr")
        print(f"      seg end {s.end:7.1f}s  {seg_lang} ({conf:.2f})", flush=True)
        for w in s.words or []:
            words.append({"start": w.start, "end": w.end, "word": w.word,
                          "lang": seg_lang, "lang_conf": conf})
    del model
    gc.collect()
    return words


def speaker_at(turns, t):
    best, dist = None, 1e9
    for s, e, spk in turns:
        if s <= t <= e:
            return spk
        d = min(abs(t - s), abs(t - e))
        if d < dist:
            best, dist = spk, d
    return best


def build_blocks(words, turns, gap=1.5):
    blocks = []
    for w in words:
        spk = speaker_at(turns, (w["start"] + w["end"]) / 2) or "SPEAKER_?"
        b = blocks[-1] if blocks else None
        if b and b["speaker"] == spk and w["start"] - b["end"] < gap:
            b["text"] += w["word"]; b["end"] = w["end"]; b["langs"].add(w["lang"])
        else:
            blocks.append({"speaker": spk, "start": w["start"], "end": w["end"],
                           "text": w["word"], "langs": {w["lang"]}})
    for b in blocks:
        b["text"] = b["text"].strip(); b["langs"] = sorted(x for x in b["langs"] if x)
    return blocks


def ts(t, srt=False):
    h, r = divmod(int(t * 1000), 3600000); m, r = divmod(r, 60000); s, ms = divmod(r, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}" if srt else f"{h:02}:{m:02}:{s:02}"


def write_outputs(out: Path, title, blocks, names, mix=None):
    out.mkdir(parents=True, exist_ok=True)
    nm = lambda s: f"{s} ({names[s]})" if s in names else s
    talk = {}
    for b in blocks:
        talk[b["speaker"]] = talk.get(b["speaker"], 0) + b["end"] - b["start"]
    md = [f"# {title}", "", "## Talk time"]
    md += [f"- {nm(s)}: {ts(d)}" for s, d in sorted(talk.items())]
    md.append("")
    if mix:
        tot = sum(mix.values()) or 1
        md += ["## Language mix (by words)"] + [f"- {l}: {100*n/tot:.0f}%" for l, n in sorted(mix.items())] + [""]
    for b in blocks:
        md += [f"[{ts(b['start'])}-{ts(b['end'])}] {nm(b['speaker'])} {'/'.join(b['langs'])}:", b["text"], ""]
    (out / "transcript.md").write_text("\n".join(md), encoding="utf-8")
    srt = []
    for i, b in enumerate(blocks, 1):
        srt += [str(i), f"{ts(b['start'], True)} --> {ts(b['end'], True)}",
                f"{names.get(b['speaker'], b['speaker'])}: {b['text']}", ""]
    (out / "transcript.srt").write_text("\n".join(srt), encoding="utf-8")
    (out / "transcript.json").write_text(json.dumps(
        {"title": title, "speakers": names, "talk_seconds": talk, "language_words": mix, "segments": blocks},
        ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--diarize-worker":
        wav, cache, n = Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
        cache.write_text(json.dumps(diarize(wav, n or None)))
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    ap.add_argument("--speakers", type=int, default=2)
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--lang", default=None, help="force tr/en; default auto per segment")
    ap.add_argument("--map", action="append", default=[], help="SPEAKER_00=Recruiter")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out-root", type=Path, default=HERE / "transcribed")
    ap.add_argument("--rep-penalty", type=float, default=1.0, help="e.g. 1.1 to discourage loops")
    ap.add_argument("--no-fallback", action="store_true", help="no temperature fallback, best_of=1")
    ap.add_argument("--no-glossary", action="store_true")
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--compute", default=None, help="int8_float16 | int8 | float16 ...")
    ap.add_argument("--redo-diar", action="store_true", help="ignore cached diarization")
    ap.add_argument("--clip", type=float, default=None, help="only first N seconds (testing)")
    a = ap.parse_args()
    EXTRA.update(best_of=1)  # fallback stays on, but cheap: best_of=5 caused OOM on 4 GB
    if a.rep_penalty != 1.0:
        EXTRA.update(repetition_penalty=a.rep_penalty)
    if a.no_fallback:
        EXTRA.update(temperature=0.0)
    names = dict(m.split("=", 1) for m in a.map)
    gloss = "" if a.no_glossary else ", ".join(
        l.strip() for l in (HERE / "glossary.txt").read_text().splitlines() if l.strip())
    out = a.out_root / f"{date.today()}-{a.name or a.audio.stem}-{a.model}"
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "a.wav"
        t0 = time.time(); to_wav(a.audio, wav)
        if a.clip:
            clipped = Path(td) / "c.wav"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-t", str(a.clip), str(clipped)], check=True)
            wav = clipped
        print(f"[1/3] diarization ...", flush=True)
        turns = diarize_cached(wav, a.speakers, out / ('diarization-clip.json' if a.clip else 'diarization.json'), a.redo_diar); print(f"      {len(turns)} turns, {time.time()-t0:.0f}s")
        print(f"[2/3] ASR {a.model} ...", flush=True)
        words = transcribe(wav, a.model, a.device, a.lang, gloss, a.beam, a.compute); print(f"      {len(words)} words, {time.time()-t0:.0f}s")
    print("[3/3] writing", out)
    mix = {}
    for w in words:
        mix[w['lang']] = mix.get(w['lang'], 0) + 1
    write_outputs(out, a.audio.stem, build_blocks(words, turns), names, mix)


if __name__ == "__main__":
    sys.exit(main())
