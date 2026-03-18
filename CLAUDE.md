# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

wisper is an audio transcription tool that extracts audio tracks from video files and transcribes them using NVIDIA NeMo ASR models (Parakeet TDT 0.6b). It produces JSON transcripts with word, segment, and character-level timestamps.

## Setup & Running

```bash
# Install dependencies (requires uv)
uv sync --extra cu128

# Run main transcription pipeline (processes all videos in video/)
python main.py

# Run single-track mode with diarization support
python main_single_track.py

# Replay a transcript with timing
python test_transcript.py
```

**System requirements:** Python 3.13, CUDA 12.8 GPU, ffmpeg/ffprobe installed.

## Architecture

**Workflow:** Place videos in `video/` → run `main.py` → audio extracted to `audio/` as mono MP3 per track → transcripts saved to `transcript/` as JSON.

**Key files:**
- `main.py` — Primary entry point. Iterates videos, extracts audio tracks via ffprobe/ffmpeg, transcribes with Parakeet TDT model, saves JSON.
- `main_single_track.py` — Alternative pipeline with speaker diarization and multitalker streaming transcription support.
- `multitalker_transcript_config.py` — Dataclass configuration for diarization and streaming ASR settings.
- `test_transcript.py` — Utility to load and replay transcripts word-by-word with timing.

**Core pipeline in `main.py`:**
1. `convert_video_to_audio_tracks()` — uses ffprobe to detect track count, ffmpeg to extract each as mono MP3
2. `transcribe_audio_parakeet()` — loads `nvidia/parakeet-tdt-0.6b-v3` model, runs `asr_model.transcribe()` with timestamps
3. `save_transcript_to_file()` — writes JSON with word/segment/char timestamp arrays

**NVIDIA models used:**
- `nvidia/parakeet-tdt-0.6b-v3` — main ASR
- `nvidia/diar_streaming_sortformer_4spk-v2.1` — speaker diarization (single-track mode)
- `nvidia/multitalker-parakeet-streaming-0.6b-v1` — multitalker ASR (single-track mode)

## Dependencies

Core: `moviepy`, `nemo-toolkit[asr]`, `whisper`. PyTorch with CUDA 12.8 via custom index (`https://download.pytorch.org/whl/cu128`).

## Notes

- `.gitignore` excludes media files (mp4, mp3, wav, avi, mkv) and JSON transcripts — these are local working data, not committed.
- Output JSON structure per track: `{ timestep, word: [{start, end, word}], segment: [{start, end, segment}], char: [{start, end, char}] }`
