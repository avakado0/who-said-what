"""One-time model download. Run by the user (reads HF_TOKEN from ./.env)."""
import os
from pathlib import Path

from huggingface_hub import snapshot_download

for line in (Path(__file__).parent / ".env").read_text().splitlines():
    if line.startswith("HF_TOKEN="):
        os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip().strip("\"'")

REPOS = [
    "pyannote/speaker-diarization-community-1",
    "pyannote/speaker-diarization-3.1",   # fallback, needs segmentation-3.0 terms accepted
    "pyannote/segmentation-3.0",
]
for repo in REPOS:
    try:
        snapshot_download(repo)
        print(repo, "OK")
    except Exception as e:
        print(repo, "FAIL", type(e).__name__)
