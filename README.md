# wisper

**wisper** is an audio transcription tool with two modes: batch file transcription and a real-time transcription service. Powered by NVIDIA NeMo ASR models (Parakeet TDT 0.6B).

## Features

- **Real-time transcription** — captures live audio from microphone and/or desktop (WASAPI loopback) and transcribes continuously
- **REST + WebSocket API** — query the latest word, sentence, last N words, or subscribe to a real-time word stream
- **Push and pull models** — poll endpoints for latest transcription, consume words from a queue, or receive words via WebSocket as they're spoken
- **Multi-track extraction** — automatically detects and extracts all audio tracks from video files via ffprobe/ffmpeg
- **NVIDIA NeMo ASR** — transcription powered by the Parakeet TDT 0.6B model for high-accuracy speech recognition
- **Speaker diarization** — optional single-track mode with streaming diarization and multitalker support
- **Flexible CLI** — process specific files, entire directories, or audio-only workflows
- **Multiple output formats** — JSON with full timestamp data or plain text
- **Batch processing** — process all files in a directory with one command

## Requirements

> **GPU required.** wisper uses NVIDIA NeMo ASR models which require a CUDA-capable GPU. It will not run on CPU.

- Python 3.13+
- NVIDIA GPU with CUDA 12.8 support and **at least 8GB VRAM** (10GB+ recommended)
- [CUDA 12.8 toolkit](https://developer.nvidia.com/cuda-12-8-0-download-archive) installed
- [ffmpeg](https://ffmpeg.org/) and ffprobe installed and on PATH
- [uv](https://github.com/astral-sh/uv) package manager
- **Windows:** [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) — required to compile the `editdistance` C extension (a dependency of `nemo-toolkit`). During installation, select the **"Desktop development with C++"** workload.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/wisper.git
   cd wisper
   ```

2. **Install dependencies (with CUDA 12.8 PyTorch):**

   ```bash
   uv sync --extra cu128
   ```

   The `cu128` extra installs PyTorch built for CUDA 12.8. This is required — there is no CPU fallback.

   > **Build error with `editdistance`?** If you see `error: Microsoft Visual C++ 14.0 or greater is required`, install the [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (select "Desktop development with C++"), then re-run `uv sync --extra cu128`.

## Quick Start

### Real-Time Service

Start the real-time transcription service:

```bash
# Transcribe from your microphone
uv run serve.py

# Transcribe desktop audio (Windows WASAPI loopback)
uv run serve.py --source desktop

# Both mic + desktop as separate streams
uv run serve.py --source both

# Both mic + desktop mixed into one stream
uv run serve.py --source both --mix mixed
```

Then query the API:

```bash
# Get the last word spoken
curl http://localhost:8000/api/last-word

# Get the last sentence
curl http://localhost:8000/api/last-segment

# Get the last 50 words
curl http://localhost:8000/api/last-words?n=50

# Consume the next word from the queue (deleted after read)
curl http://localhost:8000/api/pop-word

# Connect via WebSocket for real-time push
websocat ws://localhost:8000/ws/stream
```

When running with `--source both` (default `--mix separate`), add `?source=mic` or `?source=desktop` to any endpoint.

### Batch File Transcription

Place video files in the `video/` directory and run:

```bash
uv run main.py
```

Transcripts are saved to `transcript/` as JSON files with word, segment, and character-level timestamps.

## Usage

### Real-Time Service (`serve.py`)

```
uv run serve.py [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--source {mic,desktop,both}` | `mic` | Audio source to capture |
| `--mix {separate,mixed}` | `separate` | When `--source=both`: keep separate transcripts or mix into one |
| `--host HOST` | `127.0.0.1` | Bind address |
| `--port PORT` | `8000` | Port |
| `--chunk-seconds N` | `3.0` | Seconds of audio to accumulate before ASR inference |
| `--device-index N` | auto | Override audio device index |
| `--model NAME` | `nvidia/parakeet-tdt-0.6b-v3` | ASR model |
| `--log-level` | `INFO` | Logging level |

#### API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/status` | Service status, sources, word counts |
| `GET /api/last-word` | Most recent word |
| `GET /api/last-segment` | Most recent sentence/segment |
| `GET /api/last-words?n=100` | Last N words (rolling buffer) |
| `GET /api/pop-word` | Next word from queue (consumed on read) |
| `GET /api/pop-words?n=10` | Next N words from queue (consumed on read) |
| `WS /ws/stream` | Real-time push of words as recognized |

All endpoints accept `?source=mic` or `?source=desktop` when running in separate mode.

### Batch Transcription (`main.py`)

```
uv run main.py [OPTIONS]
```

### Input Selection

| Flag | Description |
|------|-------------|
| `--input, -i PATH` | Process a specific file or directory |
| `--audio-only` | Transcribe audio files directly, skip video extraction |
| `--tracks N [N ...]` | Only extract/transcribe specific track numbers (1-indexed) |
| `--youtube URL` | Download a YouTube video and transcribe it (cannot combine with `--input` or `--audio-only`) |

### Directories

| Flag | Default | Description |
|------|---------|-------------|
| `--video-dir PATH` | `video/` | Video input directory |
| `--audio-dir PATH` | `audio/` | Audio working directory |
| `--output-dir PATH` | `transcript/` | Transcript output directory |

### Output

| Flag | Description |
|------|-------------|
| `--format {json,txt}` | Output format (default: `json`) |
| `--model NAME` | ASR model (default: `nvidia/parakeet-tdt-0.6b-v3`) |
| `--verbose, -v` | Show segment-level timestamps during transcription |
| `--quiet, -q` | Suppress all output except errors and final file paths |

### Behavior

| Flag | Description |
|------|-------------|
| `--dry-run` | List files that would be processed without running |
| `--skip-existing` | Skip files that already have transcripts in the output directory |
| `--chunk-minutes N` | Split audio into N-minute chunks before transcribing to avoid CUDA OOM on long files (default: `5.0`). Use `0` to disable and process the full file in one pass. |

### Examples

```bash
# Process all videos in the default video/ directory
uv run main.py

# Process a single video file
uv run main.py --input video/interview.mp4

# Transcribe an existing audio file directly
uv run main.py --audio-only --input audio/podcast.mp3

# Transcribe all audio files in a custom directory
uv run main.py --audio-only --audio-dir /path/to/recordings

# Extract and transcribe only tracks 1 and 3
uv run main.py --input video/recording.mkv --tracks 1 3

# Output as plain text instead of JSON
uv run main.py --format txt

# Preview what would be processed
uv run main.py --dry-run

# Skip videos that already have transcripts
uv run main.py --skip-existing

# Verbose mode with a custom model
uv run main.py --verbose --model nvidia/parakeet-tdt-1.1b

# Download and transcribe a YouTube video
uv run main.py --youtube "https://www.youtube.com/watch?v=..."

# Transcribe a long file without chunking (caution: may OOM on large GPUs)
uv run main.py --chunk-minutes 0 --input video/longfile.mp4

# Use smaller chunks if still hitting OOM
uv run main.py --chunk-minutes 3 --input video/longfile.mp4
```

## Output Format

### JSON (default)

Each transcript is a JSON array with one entry per audio track. Each entry contains `word`, `segment`, and `char` arrays with start/end timestamps:

```json
[
  {
    "word": [{ "start": 0.0, "end": 0.32, "word": "Hello" }, ...],
    "segment": [{ "start": 0.0, "end": 2.5, "segment": "Hello world" }, ...],
    "char": [{ "start": 0.0, "end": 0.08, "char": "H" }, ...]
  }
]
```

### Plain Text (`--format txt`)

```
--- Track 1 ---
0.0s - 2.5s : Hello world
2.8s - 5.1s : This is a transcript
```

## Project Structure

```
wisper/
├── serve.py                         # Real-time service CLI entry point
├── server.py                        # FastAPI app (REST + WebSocket endpoints)
├── audio_capture.py                 # Mic + desktop audio capture (sounddevice/WASAPI)
├── realtime_asr.py                  # Chunked Parakeet ASR worker for real-time
├── transcript_store.py              # Thread-safe transcript accumulator
├── models.py                        # Shared data types (timestamps, LiveWord)
├── main.py                          # Batch CLI entry point (multi-track pipeline)
├── main_single_track.py             # Single-track pipeline with speaker diarization
├── multitalker_transcript_config.py # Configuration for diarization and streaming ASR
├── test_transcript.py               # Utility to replay transcripts with timing
├── pyproject.toml                   # Project metadata and dependencies
├── video/                           # Place input video files here
├── audio/                           # Extracted audio tracks (auto-generated)
└── transcript/                      # Output transcripts (auto-generated)
```

### Pipelines

- **`main.py`** — Multi-track pipeline. Extracts all audio tracks from video via ffprobe/ffmpeg, transcribes each with the Parakeet TDT model, and saves structured JSON or text output.
- **`main_single_track.py`** — Single-track pipeline with advanced features including streaming speaker diarization (Sortformer) and multitalker ASR for conversations with multiple speakers.

### Models

| Model | Purpose |
|-------|---------|
| `nvidia/parakeet-tdt-0.6b-v3` | Primary ASR transcription |
| `nvidia/diar_streaming_sortformer_4spk-v2.1` | Speaker diarization (single-track mode) |
| `nvidia/multitalker-parakeet-streaming-0.6b-v1` | Multitalker streaming ASR (single-track mode) |

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements and bug fixes.
