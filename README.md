# who-said-what

Offline interview transcription for M4A recordings (Turkish, English, or mixed). Produces a speaker-labelled transcript as Markdown, SRT and JSON. Audio never leaves the machine. The internet is only needed once, to download the models.

Pipeline: `ffmpeg` (M4A to 16 kHz mono WAV) -> `pyannote` speaker diarization -> `faster-whisper` `large-v3` transcription -> merge words onto speakers -> write outputs.

Tested on: Ubuntu 24.04, RTX 3050 4 GB, 30 GB RAM. A 32-minute recording takes about 2 minutes for diarization (first run only, then cached) and 5-6 minutes for transcription.

## Requirements

- NVIDIA GPU with a working driver (`nvidia-smi` shows the card). 4 GB VRAM is enough with the defaults below.
- `ffmpeg` (`sudo apt install ffmpeg`)
- Python 3.12
- A Hugging Face account with a **Read** token, and the pyannote model terms accepted (see setup)

## Setup (one time)

```bash
git clone https://github.com/avakado0/who-said-what.git
cd who-said-what
python3 -m venv .venv
source .venv/bin/activate

# torch with CUDA 12.8 wheels (also provides the cuBLAS/cuDNN libraries run.sh uses)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# keep torch pinned while installing the rest
printf 'torch==2.11.0+cu128\ntorchaudio==2.11.0+cu128\n' > /tmp/constraints.txt
pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
    -c /tmp/constraints.txt faster-whisper pyannote.audio
```

Versions used: torch 2.11.0+cu128, faster-whisper 1.2.1, pyannote.audio 4.0.7.

### Models

1. On Hugging Face, log in and click **Agree and access repository** on
   `pyannote/speaker-diarization-community-1`.
2. Create a token (type **Read**) at <https://huggingface.co/settings/tokens>. A fine-grained token also needs "Read access to contents of all public gated repos you can access".
3. Put it in `.env` in this folder (never commit it):
   ```bash
   cp .env.example .env && chmod 600 .env
   # then edit .env so the line reads HF_TOKEN=hf_...
   ```
4. Download the pyannote pipeline (the script reads `.env` itself):
   ```bash
   .venv/bin/python -I download_models.py
   ```
   `pyannote/segmentation-3.0` may print `FAIL GatedRepoError`. That is fine, `community-1` does not need it.
5. Download the Whisper model once:
   ```bash
   .venv/bin/python -I -c "from huggingface_hub import snapshot_download; snapshot_download('Systran/faster-whisper-large-v3')"
   ```

After this, `run.sh` sets `HF_HUB_OFFLINE=1` and works without a network. Run `ONLINE=1 ./run.sh ...` if a model needs downloading.

## Usage

Put recordings in `input/`, then:

```bash
./run.sh input/interview.m4a --name my_interview
```

First run without `--map`: speaker labels (`SPEAKER_00`, `SPEAKER_01`) are assigned per run in order of appearance and can swap between recordings. Read a few lines of `transcript.md`, then label them and rerun:

```bash
./run.sh input/interview.m4a --name my_interview \
    --map "SPEAKER_00=Engineering Manager" --map "SPEAKER_01=Me"
```

The second run reuses the cached diarization, so only transcription repeats.

### Options

| Flag | Default | Meaning |
|---|---|---|
| `--name` | audio filename | Name used in the output folder |
| `--speakers N` | 2 | Number of speakers. Fixing it improves diarization |
| `--lang tr\|en` | auto | Force a language. Use `tr` for all-Turkish recordings. Leave off for mixed ones |
| `--map SPEAKER_00=Name` | none | Label a speaker. Repeat for each speaker |
| `--model` | `large-v3` | Whisper model |
| `--beam N` | 5 | Beam size. Lower uses less VRAM |
| `--compute` | `int8_float16` | CTranslate2 compute type (`int8` uses less VRAM) |
| `--no-glossary` | off | Do not pass `glossary.txt` to Whisper (recommended, see Known issues) |
| `--no-fallback` | off | Disable temperature fallback. Not recommended, it lets repetition loops through |
| `--rep-penalty X` | 1.0 | Repetition penalty, e.g. 1.1 |
| `--redo-diar` | off | Ignore the cached diarization |
| `--clip SECONDS` | off | Only process the first N seconds (for quick tests) |
| `--device cuda\|cpu` | `cuda` | CPU works but is slow |
| `--out-root` | `transcribed/` | Where output folders go |

## Output

```
transcribed/<date>-<name>-<model>/
  transcript.md      readable, with timestamps, speakers, language tags, talk time, language mix
  transcript.srt     subtitles
  transcript.json    segments with start/end, speaker, languages
  diarization.json   cached speaker turns (reused on reruns)
```

The output folder is named with today's date, so the diarization cache is only found when the date matches. Rerunning on a later day re-diarizes unless you copy `diarization.json` into the new folder.

Each transcript block is tagged with a language (`tr`, `en`, or `tr/en` if it spans both). Tags come from per-segment detection that compares only Turkish against English. Segments shorter than 3 seconds, or detected with confidence below 0.85, inherit the previous segment's language.

## Known issues and design notes

- **VRAM (4 GB).** Diarization runs in its own process so its GPU memory is released before transcription. Temperature fallback is kept on (it stops repetition loops) but uses `best_of=1`, because the default `best_of=5` ran out of memory on hard segments. `CT2_CUDA_ALLOCATOR=cuda_malloc_async` is set in `run.sh` to reduce fragmentation.
- **Glossary hurts accuracy.** Passing `glossary.txt` as hotwords made Whisper drop about 30 seconds of speech in one test and hallucinate a subtitle credit line. It is **on by default**, so pass `--no-glossary` on every run. The plan is a post-transcription spelling fix list instead.
- **Speaker labels are arbitrary.** Pyannote does not know who is who and the order can differ per recording. Always check before mapping.
- **Language tags on short segments** are unreliable, hence the inherit rule above. A Turkish sentence with an English term stays `tr`; the English term is still written correctly.
- **Not yet tested** on recordings with long English passages.
- Whisper can still mishear names and rare words. Review important passages against the audio.
- `torchcodec` prints a warning at startup. It is harmless, audio is loaded without it. faster-whisper's own audio decoding is bypassed because of an incompatibility with the installed `av` version.

## Privacy

Recordings, transcripts and `.env` are listed in `.gitignore` (`input/`, `transcribed/`, `.env`) and should stay out of version control. Do not share them or paste the token anywhere.

## Files

- `transcribe.py` main pipeline and CLI
- `run.sh` wrapper: activates the venv, sets library paths, offline mode
- `download_models.py` one-time pyannote download (reads `.env`)
- `glossary.txt` technical terms for the optional hotwords prompt
