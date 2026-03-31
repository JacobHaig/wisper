import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import nemo.collections.asr as asr
from pytubefix import YouTube

from models import CharTimestamp, SegmentTimestamp, TrackTranscript, WordTimestamp

# --- Constants ---

VIDEO_DIR = Path("video")
AUDIO_DIR = Path("audio")
TRANSCRIPT_DIR = Path("transcript")
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".mp4", ".mov", ".avi", ".mkv"})
SUPPORTED_AUDIO_EXTENSIONS: frozenset[str] = frozenset({".mp3", ".wav", ".flac", ".ogg", ".m4a"})
ASR_MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"
CHUNK_OVERLAP_SECONDS: float = 15.0
DEFAULT_CHUNK_MINUTES: float = 5.0

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

    # YouTube
    parser.add_argument(
        "--youtube",
        metavar="URL",
        help="Download a YouTube video and transcribe it",
    )

    # Behavior
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed without running")
    parser.add_argument("--skip-existing", action="store_true", help="Skip files that already have transcripts")
    parser.add_argument(
        "--chunk-minutes",
        type=float,
        default=DEFAULT_CHUNK_MINUTES,
        metavar="N",
        help=f"Split audio into N-minute chunks to avoid CUDA OOM on long files (default: {DEFAULT_CHUNK_MINUTES}). Use 0 to disable.",
    )

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


# --- YouTube download ---

def download_youtube_video(url: str, video_dir: Path) -> Path:
    """Download a YouTube video to video_dir and return the path to the file."""
    video_dir.mkdir(parents=True, exist_ok=True)
    yt = YouTube(url, on_progress_callback=lambda stream, chunk, remaining: (
        _print(f"  Downloading... {100 - round(remaining / stream.filesize * 100)}%", level=2)
    ))
    _print(f"  Title: {yt.title}")
    stream = yt.streams.get_highest_resolution()
    out_path = stream.download(output_path=str(video_dir))
    return Path(out_path)


# --- Audio chunking ---

def get_audio_duration_seconds(audio_path: Path) -> float:
    """Return the duration of an audio file in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json",
            str(audio_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])


def split_audio_into_chunks(
    audio_path: Path,
    chunk_seconds: float,
    overlap_seconds: float,
    temp_dir: Path,
) -> list[tuple[Path, float]]:
    """
    Split an audio file into overlapping chunks using ffmpeg.
    Returns a list of (chunk_path, offset_seconds) pairs.
    If the file is shorter than chunk_seconds, returns [(audio_path, 0.0)] with no splitting.
    """
    duration = get_audio_duration_seconds(audio_path)

    if duration <= chunk_seconds:
        return [(audio_path, 0.0)]

    stride = chunk_seconds - overlap_seconds
    chunks: list[tuple[Path, float]] = []
    start = 0.0
    idx = 0

    while start < duration:
        chunk_dur = min(chunk_seconds, duration - start)
        chunk_path = temp_dir / f"chunk_{idx:03d}.mp3"
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", str(start),
                "-t", str(chunk_dur),
                "-i", str(audio_path),
                "-c", "copy",
                str(chunk_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        _print(f"  Chunk {idx + 1}: {start:.1f}s – {start + chunk_dur:.1f}s", level=2)
        chunks.append((chunk_path, start))
        start += stride
        idx += 1

    return chunks


def merge_chunk_transcripts(
    chunk_results: list[tuple[TrackTranscript, float]],
    overlap_seconds: float,
) -> TrackTranscript:
    """
    Merge per-chunk transcripts into one continuous TrackTranscript.
    Timestamps are shifted by each chunk's offset. The midpoint of each
    overlap window acts as the cut line — items before the midpoint come
    from the earlier chunk, items from the midpoint onwards from the later chunk.
    """
    if len(chunk_results) == 1:
        return chunk_results[0][0]

    # Cut line between chunk i and chunk i+1: midpoint of their overlap
    cuts: list[float] = []
    for i in range(len(chunk_results) - 1):
        next_offset = chunk_results[i + 1][1]
        cuts.append(next_offset + overlap_seconds / 2)

    all_words: list[WordTimestamp] = []
    all_segments: list[SegmentTimestamp] = []
    all_chars: list[CharTimestamp] = []

    for i, (transcript, offset) in enumerate(chunk_results):
        lower = cuts[i - 1] if i > 0 else 0.0
        upper = cuts[i] if i < len(cuts) else float("inf")

        for w in transcript.word:
            adj_start = w.start + offset
            if lower <= adj_start < upper:
                all_words.append(WordTimestamp(start=adj_start, end=w.end + offset, word=w.word))

        for s in transcript.segment:
            adj_start = s.start + offset
            if lower <= adj_start < upper:
                all_segments.append(SegmentTimestamp(start=adj_start, end=s.end + offset, segment=s.segment))

        for c in transcript.char:
            adj_start = c.start + offset
            if lower <= adj_start < upper:
                all_chars.append(CharTimestamp(start=adj_start, end=c.end + offset, char=c.char))

        _print(f"  Chunk {i + 1}: kept {len([w for w in transcript.word if lower <= w.start + offset < upper])} words in [{lower:.1f}s, {'∞' if upper == float('inf') else f'{upper:.1f}s'})", level=2)

    return TrackTranscript(
        word=sorted(all_words, key=lambda x: x.start),
        segment=sorted(all_segments, key=lambda x: x.start),
        char=sorted(all_chars, key=lambda x: x.start),
    )


# --- Pipeline functions ---

def _output_to_transcript(output) -> TrackTranscript:
    """Convert a NeMo ASR output to a TrackTranscript."""
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
    return transcript


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


def transcribe_audio_parakeet(
    audio_paths: list[Path],
    model_name: str = ASR_MODEL_NAME,
    chunk_seconds: float | None = None,
) -> list[TrackTranscript]:
    """Transcribe audio files using NVIDIA Parakeet TDT model.

    If chunk_seconds is set, each audio file is split into overlapping chunks
    before transcription and the results are merged into one continuous transcript.
    """
    _print(f"Loading ASR model: {model_name}")
    asr_model = asr.models.ASRModel.from_pretrained(model_name=model_name)

    transcripts: list[TrackTranscript] = []

    for audio_path in audio_paths:
        _print(f"Transcribing: {audio_path.name}")

        if chunk_seconds is None:
            # Original one-shot path
            outputs = asr_model.transcribe([str(audio_path)], timestamps=True)
            transcripts.append(_output_to_transcript(outputs[0]))
            continue

        # Chunked path
        with tempfile.TemporaryDirectory() as tmp:
            chunks = split_audio_into_chunks(audio_path, chunk_seconds, CHUNK_OVERLAP_SECONDS, Path(tmp))

            if len(chunks) == 1 and chunks[0][0] == audio_path:
                # File is shorter than chunk size — transcribe directly
                outputs = asr_model.transcribe([str(audio_path)], timestamps=True)
                transcripts.append(_output_to_transcript(outputs[0]))
                continue

            n = len(chunks)
            _print(f"  Split into {n} chunk(s) of {chunk_seconds / 60:.1f} min (overlap: {CHUNK_OVERLAP_SECONDS}s)")

            chunk_results: list[tuple[TrackTranscript, float]] = []
            for i, (chunk_path, offset) in enumerate(chunks):
                _print(f"  Transcribing chunk {i + 1}/{n} (offset {offset:.1f}s)...")
                outputs = asr_model.transcribe([str(chunk_path)], timestamps=True)
                chunk_results.append((_output_to_transcript(outputs[0]), offset))

            _print(f"  Merging {n} chunk transcripts...")
            transcripts.append(merge_chunk_transcripts(chunk_results, CHUNK_OVERLAP_SECONDS))

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


def video_to_transcript(
    video_path: Path,
    audio_dir: Path,
    transcript_output_path: Path,
    chunk_seconds: float | None = None,
) -> None:
    """Full pipeline: video -> audio tracks -> transcription -> JSON."""
    audio_paths = convert_video_to_audio_tracks(video_path, audio_dir)
    transcripts = transcribe_audio_parakeet(audio_paths, chunk_seconds=chunk_seconds)
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

    if args.chunk_minutes < 0:
        print("Error: --chunk-minutes must be >= 0 (use 0 to disable chunking)", file=sys.stderr)
        sys.exit(1)
    if 0 < args.chunk_minutes < 0.5:
        print("Error: --chunk-minutes must be >= 0.5 or 0 to disable", file=sys.stderr)
        sys.exit(1)

    chunk_seconds: float | None = args.chunk_minutes * 60.0 if args.chunk_minutes > 0 else None

    audio_dir: Path = args.audio_dir
    output_dir: Path = args.output_dir

    if args.youtube:
        if args.input or args.audio_only:
            print("Error: --youtube cannot be combined with --input or --audio-only", file=sys.stderr)
            sys.exit(1)
        if args.dry_run:
            print(f"Would download and transcribe: {args.youtube}")
            return
        _print(f"Downloading: {args.youtube}")
        input_files = [download_youtube_video(args.youtube, args.video_dir)]
    else:
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

        transcripts = transcribe_audio_parakeet(audio_paths, model_name=args.model, chunk_seconds=chunk_seconds)

        if args.format == "txt":
            save_transcript_as_text(transcripts, output_path)
        else:
            save_transcript_to_file(transcripts, output_path)


if __name__ == "__main__":
    main()
