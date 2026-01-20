

# wisper

**wisper** is an audio transcription tool designed to convert audio files into accurate, readable text. It is lightweight, easy to use, and leverages modern Python tooling for fast setup and reproducible environments.

## Features
- Transcribe audio files to text with high accuracy
- Batch processing for multiple files
- Configurable output formats
- Simple command-line interface
- Fast environment setup using [uv](https://github.com/astral-sh/uv)

## Installation

1. **Clone the repository:**
	```bash
	git clone https://github.com/yourusername/wisper.git
	cd wisper
	```

2. **Install dependencies with [uv](https://github.com/astral-sh/uv):**

	Using `uv` with a `pyproject.toml`:
	```bash
	uv sync --extra cu128
	```

## Usage

Run the main script to transcribe audio files:

```bash
uv pip install .  # if you want to install as a package
python main.py --input audio/yourfile.wav --output transcript/yourfile.json
```

- Replace `audio/yourfile.wav` with your audio file path.
- The transcript will be saved in the specified output location.

## Configuration

You can adjust transcription settings in `multitalker_transcript_config.py` or use the provided JSON config files.

## Project Structure
- `main.py` — Main entry point for transcription
- `audio/` — Place your audio files here
- `transcript/` — Transcription outputs
- `multitalker_transcript_config.py` — Configuration options
- `pyproject.toml` — Project metadata and dependencies

## Contributing
Contributions are welcome! Please open issues or submit pull requests for improvements and bug fixes.


