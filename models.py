"""Shared data types for wisper transcription pipelines."""

from dataclasses import asdict, dataclass


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


@dataclass(frozen=True, slots=True)
class LiveWord:
    """A single recognized word from real-time transcription."""
    timestamp: float   # seconds since capture start
    word: str
    source: str        # "mic" | "desktop" | "mixed"
