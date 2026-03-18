import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import nemo.collections.asr as asr

# --- Constants ---

VIDEO_DIR = Path("video")
AUDIO_DIR = Path("audio")
TRANSCRIPT_DIR = Path("transcript")
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".mp4", ".mov", ".avi", ".mkv"})
SUPPORTED_AUDIO_EXTENSIONS: frozenset[str] = frozenset({".mp3", ".wav", ".flac", ".ogg", ".m4a"})
ASR_MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"

# --- Verbosity ---

_verbosity: int = 1  # 0=quiet, 1=normal, 2=verbose


def _print(msg: str, *, level: int = 1) -> None:
    if _verbosity >= level:
        print(msg)


# --- CLI ---

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="wisper — extract audio from video and transcribe with NVIDIA NeMo ASR",
    )

    # Input selection
    parser.add_argument(
        "--input", "-i",
        type=Path,
        help="Process a specific file or directory (default: all files in --video-dir)",
    )
    parser.add_argument(
        "--audio-only",
        action="store_true",
        help="Transcribe audio files directly, skip video extraction",
    )
    parser.add_argument(
        "--tracks",
        type=int,
        nargs="+",
        metavar="N",
        help="Only extract/transcribe specific track numbers (1-indexed)",
    )

    # Directories
    parser.add_argument("--video-dir", type=Path, default=VIDEO_DIR, help="Video input directory (default: video/)")
    parser.add_argument("--audio-dir", type=Path, default=AUDIO_DIR, help="Audio working directory (default: audio/)")
    parser.add_argument("--output-dir", type=Path, default=TRANSCRIPT_DIR, help="Transcript output directory (default: transcript/)")

    # Model
    parser.add_argument("--model", default=ASR_MODEL_NAME, help=f"ASR model name (default: {ASR_MODEL_NAME})")

    # Output format
    parser.add_argument("--format", choices=["json", "txt"], default="json", help="Output format (default: json)")

    # Verbosity (mutually exclusive)
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("--verbose", "-v", action="store_true", help="Show segment-level timestamps during transcription")
    verbosity.add_argument("--quiet", "-q", action="store_true", help="Suppress all output except errors and final path")

    # Behavior
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed without running")
    parser.add_argument("--skip-existing", action="store_true", help="Skip files that already have transcripts")

    return parser


def resolve_input_files(args: argparse.Namespace) -> list[Path]:
    """Determine which files to process based on CLI arguments."""
    extensions = SUPPORTED_AUDIO_EXTENSIONS if args.audio_only else SUPPORTED_EXTENSIONS

    if args.input is not None:
        path = args.input
        if not path.exists():
            print(f"Error: {path} does not exist", file=sys.stderr)
            sys.exit(1)

        if path.is_file():
            if path.suffix.lower() not in extensions:
                print(
                    f"Error: unsupported file type '{path.suffix}'. "
                    f"Expected: {', '.join(sorted(extensions))}",
                    file=sys.stderr,
                )
                sys.exit(1)
            return [path]

        if path.is_dir():
            files = sorted(p for p in path.iterdir() if p.suffix.lower() in extensions)
            if not files:
                print(f"Error: no supported files found in {path}", file=sys.stderr)
                sys.exit(1)
            return files

    # No --input: scan default directory
    scan_dir = args.audio_dir if args.audio_only else args.video_dir
    if not scan_dir.exists():
        print(f"Error: directory {scan_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    files = sorted(p for p in scan_dir.iterdir() if p.suffix.lower() in extensions)
    if not files:
        print(f"Error: no supported files found in {scan_dir}", file=sys.stderr)
        sys.exit(1)
    return files


# --- Typed transcript structures ---

@dataclass(frozen=True, slots=True)
class WordTimestamp:
    start: float
    end: float
    word: str


@dataclass(frozen=True, slots=True)
class SegmentTimestamp:
    start: float
    end: float
    segment: str


@dataclass(frozen=True, slots=True)
class CharTimestamp:
    start: float
    end: float
    char: str


@dataclass(frozen=True, slots=True)
class TrackTranscript:
    word: list[WordTimestamp]
    segment: list[SegmentTimestamp]
    char: list[CharTimestamp]

    def to_dict(self) -> dict:
        return {
            "word": [asdict(w) for w in self.word],
            "segment": [asdict(s) for s in self.segment],
            "char": [asdict(c) for c in self.char],
        }


# --- Pipeline functions ---

def convert_video_to_audio_tracks(video_path: Path, audio_dir: Path) -> list[Path]:
    """Extract all audio tracks from a video file as mono MP3s."""
    audio_dir.mkdir(parents=True, exist_ok=True)

    _print(f"Getting number of audio tracks in {video_path}...")
    probe_result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "a",
            "-show_entries", "stream=index", "-of", "json",
            str(video_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    info: dict = json.loads(probe_result.stdout)
    streams: list[dict] = info.get("streams", [])

    if not streams:
        raise RuntimeError(f"No audio streams found in {video_path}")

    _print(f"Found {len(streams)} audio track(s). Exporting each as mono...")

    stem = video_path.stem
    audio_paths: list[Path] = []
    for i in range(len(streams)):
        track_output = audio_dir / f"{stem}_track{i + 1}.mp3"
        _print(f"Extracting track {i + 1} to {track_output}...")
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(video_path),
                "-map", f"0:a:{i}", "-ac", "1",
                "-codec:a", "libmp3lame", str(track_output),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        _print(f"Exported {track_output}")
        audio_paths.append(track_output)

    _print("All audio tracks exported.")
    return audio_paths


def transcribe_audio_parakeet(audio_paths: list[Path], model_name: str = ASR_MODEL_NAME) -> list[TrackTranscript]:
    """Transcribe audio files using NVIDIA Parakeet TDT model."""
    str_paths = [str(p) for p in audio_paths]
    _print(f"Performing transcription on: {str_paths}")

    asr_model = asr.models.ASRModel.from_pretrained(model_name=model_name)
    outputs = asr_model.transcribe(str_paths, timestamps=True)

    transcripts: list[TrackTranscript] = []
    for output in outputs:
        ts = output.timestamp
        transcript = TrackTranscript(
            word=[WordTimestamp(start=w["start"], end=w["end"], word=w["word"]) for w in ts["word"]],
            segment=[SegmentTimestamp(start=s["start"], end=s["end"], segment=s["segment"]) for s in ts["segment"]],
            char=[CharTimestamp(start=c["start"], end=c["end"], char=c["char"]) for c in ts["char"]],
        )

        if transcript.segment:
            _print("\nSegment-level Timestamps:", level=2)
            for seg in transcript.segment:
                _print(f"  {round(seg.start, 2)}s - {round(seg.end, 2)}s : {seg.segment}", level=2)

        transcripts.append(transcript)

    return transcripts


def save_transcript_to_file(transcripts: list[TrackTranscript], output_path: Path) -> None:
    """Serialize transcripts to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = [t.to_dict() for t in transcripts]
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    _print(f"Transcript saved to {output_path}", level=0)


def save_transcript_as_text(transcripts: list[TrackTranscript], output_path: Path) -> None:
    """Serialize transcripts to plain text with timestamps."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for i, transcript in enumerate(transcripts):
        if len(transcripts) > 1:
            lines.append(f"--- Track {i + 1} ---")
        for seg in transcript.segment:
            lines.append(f"{round(seg.start, 2)}s - {round(seg.end, 2)}s : {seg.segment}")
        if len(transcripts) > 1:
            lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    _print(f"Transcript saved to {output_path}", level=0)


def video_to_transcript(video_path: Path, audio_dir: Path, transcript_output_path: Path) -> None:
    """Full pipeline: video -> audio tracks -> transcription -> JSON."""
    audio_paths = convert_video_to_audio_tracks(video_path, audio_dir)
    transcripts = transcribe_audio_parakeet(audio_paths)
    save_transcript_to_file(transcripts, transcript_output_path)


# --- Main ---

def main() -> None:
    global _verbosity
    parser = build_parser()
    args = parser.parse_args()

    if args.quiet:
        _verbosity = 0
    elif args.verbose:
        _verbosity = 2
    else:
        _verbosity = 1

    audio_dir: Path = args.audio_dir
    output_dir: Path = args.output_dir
    input_files = resolve_input_files(args)

    if args.dry_run:
        print(f"Would process {len(input_files)} file(s):")
        for f in input_files:
            print(f"  {f}")
        return

    for filepath in input_files:
        ext = ".txt" if args.format == "txt" else ".json"
        output_path = output_dir / f"{filepath.stem}{ext}"

        if args.skip_existing and output_path.exists():
            _print(f"Skipping {filepath.name} (transcript exists)")
            continue

        _print(f"Processing: {filepath.name}")

        if args.audio_only:
            audio_paths = [filepath]
            if args.tracks:
                _print("Warning: --tracks is ignored in --audio-only mode", level=0)
        else:
            audio_paths = convert_video_to_audio_tracks(filepath, audio_dir)
            if args.tracks:
                requested = [t - 1 for t in args.tracks]
                filtered = [audio_paths[i] for i in requested if 0 <= i < len(audio_paths)]
                skipped = [t for t in args.tracks if t - 1 < 0 or t - 1 >= len(audio_paths)]
                if skipped:
                    _print(f"Warning: track(s) {skipped} out of range (1-{len(audio_paths)})", level=0)
                if not filtered:
                    _print(f"Error: no valid tracks selected for {filepath.name}", level=0)
                    continue
                audio_paths = filtered

        transcripts = transcribe_audio_parakeet(audio_paths, model_name=args.model)

        if args.format == "txt":
            save_transcript_as_text(transcripts, output_path)
        else:
            save_transcript_to_file(transcripts, output_path)


if __name__ == "__main__":
    main()
