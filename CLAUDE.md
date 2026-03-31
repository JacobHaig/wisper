# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

wisper is an audio transcription tool with three modes:
1. **Batch CLI** — extracts audio tracks from video files and transcribes them using NVIDIA NeMo ASR models (Parakeet TDT 0.6b). Produces JSON transcripts with word, segment, and character-level timestamps.
2. **Real-time service** — captures live audio from microphone, desktop (WASAPI loopback), or a network audio stream (RTP/RTSP/UDP) and transcribes continuously, serving results via a FastAPI REST + WebSocket API.
3. **Single-track with diarization** — alternative pipeline with streaming speaker diarization (Sortformer) for conversations with multiple speakers.

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

# --- Real-time transcription service ---

# Start with microphone (default)
uv run serve.py

# Start with desktop audio (WASAPI loopback, Windows only)
uv run serve.py --source desktop

# Both mic + desktop as separate streams
uv run serve.py --source both

# Both mic + desktop mixed into one stream
uv run serve.py --source both --mix mixed

# Network audio stream (RTP/RTSP/UDP) via ffmpeg
uv run serve.py --source network --stream-url rtp://0.0.0.0:5004
uv run serve.py --stream-url rtsp://192.168.1.10/live   # auto-sets --source=network

# Custom port and chunk size
uv run serve.py --port 9000 --chunk-seconds 5.0

# API endpoints (while server is running):
# GET  http://localhost:8000/api/status
# GET  http://localhost:8000/api/last-word
# GET  http://localhost:8000/api/last-segment
# GET  http://localhost:8000/api/last-words?n=100
# GET  http://localhost:8000/api/pop-word          (consumed on read)
# GET  http://localhost:8000/api/pop-words?n=10    (consumed on read)
# WS   ws://localhost:8000/ws/stream               (real-time push)
# For separate mode, add ?source=mic or ?source=desktop to any endpoint
```

**System requirements:** Python 3.13, CUDA 12.8 GPU, ffmpeg/ffprobe installed.

## Architecture

**Workflow:** Place videos in `video/` → run `main.py` → audio extracted to `audio/` as mono MP3 per track → transcripts saved to `transcript/` as JSON or plain text.

**Key files:**
- `main.py` — Primary CLI entry point for batch transcription. Full argparse interface with input selection (`--input`, `--audio-only`, `--tracks`), output format (`--format json|txt`), directory overrides, verbosity control, `--skip-existing`, and `--dry-run`. Extracts audio via ffprobe/ffmpeg, transcribes with Parakeet TDT, saves structured output.
- `serve.py` — CLI entry point for the real-time transcription service. Starts audio capture + ASR workers + FastAPI server.
- `server.py` — FastAPI application with REST and WebSocket endpoints for querying live transcription.
- `models.py` — Shared data types (`WordTimestamp`, `SegmentTimestamp`, `CharTimestamp`, `TrackTranscript`, `LiveWord`) used by both batch and real-time pipelines.
- `audio_capture.py` — Audio source abstraction. Supports mic (sounddevice), desktop (WASAPI loopback), or network stream (ffmpeg subprocess piping PCM to the queue). All sources produce the same float32/16kHz chunks regardless of origin.
- `realtime_asr.py` — Chunked Parakeet TDT worker that reads audio from a queue, runs inference, deduplicates overlap, and writes results to a `TranscriptStore`.
- `transcript_store.py` — Thread-safe accumulator for live transcription results. Rolling word/segment buffer, HTTP pop queue, and WebSocket subscriber queues.
- `main_single_track.py` — Alternative pipeline with speaker diarization (Sortformer) and multitalker streaming transcription support. Uses moviepy for audio extraction.
- `multitalker_transcript_config.py` — Dataclass configuration for diarization and streaming ASR settings.
- `test_transcript.py` — Utility to load and replay transcripts word-by-word with timing. Has a hardcoded input path; update `input_path` in `main()` before use.

**Core pipeline in `main.py`:**
1. `resolve_input_files()` — determines files from `--input`, `--audio-only`, or default directory scan
2. `convert_video_to_audio_tracks()` — uses ffprobe to detect track count, ffmpeg to extract each as mono MP3
3. `transcribe_audio_parakeet()` — loads `nvidia/parakeet-tdt-0.6b-v3` model; if `--chunk-minutes > 0`, splits each audio file into overlapping chunks via ffmpeg, transcribes each independently, then merges into one continuous `TrackTranscript`; otherwise transcribes in one pass
4. `save_transcript_to_file()` — writes JSON with word/segment/char timestamp arrays
5. `save_transcript_as_text()` — writes plain text with `start - end : segment` lines (used when `--format txt`)

**Audio chunking** (`--chunk-minutes`, default `5.0`): Long audio files are split into overlapping chunks using ffmpeg before transcription to avoid CUDA OOM. Each chunk is transcribed independently and the results are merged back into one single transcript — the output is identical in structure to a non-chunked run. The overlap window (15s, hardcoded as `CHUNK_OVERLAP_SECONDS`) prevents words being cut at boundaries. Use `--chunk-minutes 0` to disable chunking entirely.

**NVIDIA models used:**
- `nvidia/parakeet-tdt-0.6b-v3` — main ASR
- `nvidia/diar_streaming_sortformer_4spk-v2.1` — speaker diarization (single-track mode)
- `nvidia/multitalker-parakeet-streaming-0.6b-v1` — multitalker ASR (single-track mode)

**Real-time service architecture:**
- Audio capture → `queue.Queue[np.ndarray]` (float32, 16kHz mono)
  - **mic/desktop**: sounddevice callback with on-the-fly resampling
  - **network**: ffmpeg subprocess piping PCM via stdout, read in a background thread
- ASR worker thread (reads queue, runs Parakeet TDT on 3s chunks with 1s overlap, deduplicates) → `TranscriptStore`
- FastAPI/uvicorn (async HTTP + WebSocket, reads from store)

**Real-time API endpoints:**
- `GET /api/status` — service status, sources, word counts
- `GET /api/last-word` — most recent word
- `GET /api/last-segment` — most recent sentence/segment
- `GET /api/last-words?n=100` — last N words (rolling buffer)
- `GET /api/pop-word` — consume next word from queue (deleted after read)
- `GET /api/pop-words?n=10` — consume N words from queue
- `WS /ws/stream` — real-time push of words as they're recognized
- All endpoints accept `?source=mic|desktop` when running in separate mode

## Dependencies

Core: `moviepy`, `nemo-toolkit[asr]`, `pytubefix`, `fastapi`, `uvicorn`, `sounddevice`, `soundfile`, `scipy`. PyTorch with CUDA 12.8 via custom index (`https://download.pytorch.org/whl/cu128`).

## Notes

- `.gitignore` excludes media files (mp4, mp3, wav, avi, mkv) and JSON transcripts — these are local working data, not committed.
- Output JSON structure per track: `{ word: [{start, end, word}], segment: [{start, end, segment}], char: [{start, end, char}] }`


## For Claude

Each time we complete the changes, we need to use 'uv run ...' to test and validate the changes worked.
When making changes, please ensure that the code is well-structured, follows best practices, and includes appropriate error handling. 
If new information is needed to complete the task, please ask for clarification before proceeding.
After completing the changes, please consider making changes to CLAUDE.md and or README.md to reflect the changes made and to provide clear documentation for future reference.
