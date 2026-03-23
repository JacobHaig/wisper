# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

wisper is an audio transcription tool that extracts audio tracks from video files and transcribes them using NVIDIA NeMo ASR models (Parakeet TDT 0.6b). It produces JSON transcripts with word, segment, and character-level timestamps.

## Setup & Running

```bash
# Install dependencies (requires uv)
uv sync --extra cu128

# Run main transcription pipeline (processes all videos in video/)
uv run main.py

# Common CLI flags
uv run main.py --input video/interview.mp4        # single file
uv run main.py --audio-only --input audio/pod.mp3 # skip video extraction
uv run main.py --format txt                        # plain text output
uv run main.py --skip-existing --dry-run           # preview without running

# Run single-track mode with diarization support
uv run main_single_track.py

# Replay a transcript with timing (edit hardcoded path in file first)
uv run test_transcript.py
```

**System requirements:** Python 3.13, CUDA 12.8 GPU, ffmpeg/ffprobe installed.

## Architecture

**Workflow:** Place videos in `video/` → run `main.py` → audio extracted to `audio/` as mono MP3 per track → transcripts saved to `transcript/` as JSON or plain text.

**Key files:**
- `main.py` — Primary CLI entry point. Full argparse interface with input selection (`--input`, `--audio-only`, `--tracks`), output format (`--format json|txt`), directory overrides, verbosity control, `--skip-existing`, and `--dry-run`. Extracts audio via ffprobe/ffmpeg, transcribes with Parakeet TDT, saves structured output.
- `main_single_track.py` — Alternative pipeline with speaker diarization (Sortformer) and multitalker streaming transcription support. Uses moviepy for audio extraction.
- `multitalker_transcript_config.py` — Dataclass configuration for diarization and streaming ASR settings.
- `test_transcript.py` — Utility to load and replay transcripts word-by-word with timing. Has a hardcoded input path; update `input_path` in `main()` before use.

**Core pipeline in `main.py`:**
1. `resolve_input_files()` — determines files from `--input`, `--audio-only`, or default directory scan
2. `convert_video_to_audio_tracks()` — uses ffprobe to detect track count, ffmpeg to extract each as mono MP3
3. `transcribe_audio_parakeet()` — loads `nvidia/parakeet-tdt-0.6b-v3` model, runs `asr_model.transcribe()` with timestamps; returns typed `TrackTranscript` dataclasses
4. `save_transcript_to_file()` — writes JSON with word/segment/char timestamp arrays
5. `save_transcript_as_text()` — writes plain text with `start - end : segment` lines (used when `--format txt`)

**NVIDIA models used:**
- `nvidia/parakeet-tdt-0.6b-v3` — main ASR
- `nvidia/diar_streaming_sortformer_4spk-v2.1` — speaker diarization (single-track mode)
- `nvidia/multitalker-parakeet-streaming-0.6b-v1` — multitalker ASR (single-track mode)

## Dependencies

Core: `moviepy`, `nemo-toolkit[asr]`, `yt-dlp`. PyTorch with CUDA 12.8 via custom index (`https://download.pytorch.org/whl/cu128`).

## Notes

- `.gitignore` excludes media files (mp4, mp3, wav, avi, mkv) and JSON transcripts — these are local working data, not committed.
- Output JSON structure per track: `{ word: [{start, end, word}], segment: [{start, end, segment}], char: [{start, end, char}] }`


## For Claude

Each time we complete the changes, we need to use 'uv run ...' to test and validate the changes worked.
When making changes, please ensure that the code is well-structured, follows best practices, and includes appropriate error handling. 
If new information is needed to complete the task, please ask for clarification before proceeding.
After completing the changes, please consider making changes to CLAUDE.md and or README.md to reflect the changes made and to provide clear documentation for future reference.
