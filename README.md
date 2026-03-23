# wisper

**wisper** is an audio transcription tool that extracts audio tracks from video files and transcribes them using NVIDIA NeMo ASR models. It produces structured transcripts with word, segment, and character-level timestamps.

## Features

- **Multi-track extraction** — automatically detects and extracts all audio tracks from video files via ffprobe/ffmpeg
- **NVIDIA NeMo ASR** — transcription powered by the Parakeet TDT 0.6B model for high-accuracy speech recognition
- **Speaker diarization** — optional single-track mode with streaming diarization and multitalker support
- **Flexible CLI** — process specific files, entire directories, or audio-only workflows
- **Multiple output formats** — JSON with full timestamp data or plain text
- **Batch processing** — process all files in a directory with one command

## Requirements

- Python 3.13+
- CUDA 12.8 compatible GPU
- [ffmpeg](https://ffmpeg.org/) and ffprobe installed and on PATH
- [uv](https://github.com/astral-sh/uv) (recommended package manager)

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/wisper.git
   cd wisper
   ```

2. **Install dependencies:**

   ```bash
   uv sync --extra cu128
   ```

## Quick Start

Place video files in the `video/` directory and run:

```bash
uv run main.py
```

Transcripts are saved to `transcript/` as JSON files with word, segment, and character-level timestamps.

## Usage

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
├── main.py                          # Primary CLI entry point (multi-track pipeline)
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
