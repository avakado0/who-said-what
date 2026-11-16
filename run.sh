#!/usr/bin/env bash
# Usage: ./run.sh path/to/recording.m4a [--speakers 2] [--model large-v3] [--map SPEAKER_01=Candidate]
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
SITE="$(python -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')"
export LD_LIBRARY_PATH="$SITE/nvidia/cublas/lib:$SITE/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
# pyannote telemetry is ON by default; disable (no network calls).
export PYANNOTE_METRICS_ENABLED=false
export CT2_CUDA_ALLOCATOR=cuda_malloc_async  # less fragmentation -> fewer OOMs on 4 GB
# Offline after models are cached. Set ONLINE=1 to allow downloads.
if [ -z "$ONLINE" ]; then export HF_HUB_OFFLINE=1; fi
python -I transcribe.py "$@"
