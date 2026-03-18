import json
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

import nemo.collections.asr as asr

# --- Constants ---

VIDEO_DIR = Path("video")
AUDIO_DIR = Path("audio")
TRANSCRIPT_DIR = Path("transcript")
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".mp4", ".mov", ".avi", ".mkv"})
ASR_MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"


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

    print(f"Getting number of audio tracks in {video_path}...")
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

    print(f"Found {len(streams)} audio track(s). Exporting each as mono...")

    stem = video_path.stem
    audio_paths: list[Path] = []
    for i in range(len(streams)):
        track_output = audio_dir / f"{stem}_track{i + 1}.mp3"
        print(f"Extracting track {i + 1} to {track_output}...")
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
        print(f"Exported {track_output}")
        audio_paths.append(track_output)

    print("All audio tracks exported.")
    return audio_paths


def transcribe_audio_parakeet(audio_paths: list[Path]) -> list[TrackTranscript]:
    """Transcribe audio files using NVIDIA Parakeet TDT model."""
    str_paths = [str(p) for p in audio_paths]
    print(f"Performing transcription on: {str_paths}")

    asr_model = asr.models.ASRModel.from_pretrained(model_name=ASR_MODEL_NAME)
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
            print("\nSegment-level Timestamps:")
            for seg in transcript.segment:
                print(f"  {round(seg.start, 2)}s - {round(seg.end, 2)}s : {seg.segment}")

        transcripts.append(transcript)

    return transcripts


def save_transcript_to_file(transcripts: list[TrackTranscript], output_path: Path) -> None:
    """Serialize transcripts to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = [t.to_dict() for t in transcripts]
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Transcript saved to {output_path}")


def video_to_transcript(video_path: Path, audio_dir: Path, transcript_output_path: Path) -> None:
    """Full pipeline: video -> audio tracks -> transcription -> JSON."""
    audio_paths = convert_video_to_audio_tracks(video_path, audio_dir)
    transcripts = transcribe_audio_parakeet(audio_paths)
    save_transcript_to_file(transcripts, transcript_output_path)


def main() -> None:
    for entry in VIDEO_DIR.iterdir():
        if entry.suffix not in SUPPORTED_EXTENSIONS:
            continue

        print(f"Processing video: {entry.name}")

        audio_dir = AUDIO_DIR
        transcript_output = TRANSCRIPT_DIR / f"{entry.stem}.json"

        print(f"Video Path: {entry}")
        print(f"Audio Dir: {audio_dir}")
        print(f"Transcript Output Path: {transcript_output}")

        video_to_transcript(entry, audio_dir, transcript_output)


if __name__ == "__main__":
    main()
