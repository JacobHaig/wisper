"""Real-time ASR worker using chunked Parakeet TDT inference."""

from __future__ import annotations

import logging
import queue
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from models import LiveWord
from transcript_store import TranscriptStore

logger = logging.getLogger(__name__)

# Default ASR model
ASR_MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"


class ChunkedParakeetWorker:
    """Accumulates audio chunks, periodically runs Parakeet TDT, emits words.

    The worker reads float32/16kHz audio chunks from an input queue,
    accumulates them into a buffer, and when the buffer reaches
    ``chunk_seconds`` of audio it writes a temp WAV, runs model inference,
    deduplicates against previously emitted words, and pushes new words
    into the TranscriptStore.

    Parameters
    ----------
    store : TranscriptStore
        Where to write recognised words and segments.
    source_label : str
        Label for the audio source ("mic", "desktop", or "mixed").
    model_name : str
        HuggingFace / NeMo model identifier.
    chunk_seconds : float
        How many seconds of audio to accumulate before running inference.
    overlap_seconds : float
        Seconds of audio kept from the previous chunk to avoid cutting words
        at boundaries.
    sample_rate : int
        Expected sample rate of incoming audio.
    """

    def __init__(
        self,
        store: TranscriptStore,
        source_label: str = "mic",
        model_name: str = ASR_MODEL_NAME,
        chunk_seconds: float = 3.0,
        overlap_seconds: float = 1.0,
        sample_rate: int = 16_000,
    ) -> None:
        self.store = store
        self.source_label = source_label
        self.model_name = model_name
        self.chunk_seconds = chunk_seconds
        self.overlap_seconds = overlap_seconds
        self.sample_rate = sample_rate

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._audio_queue: queue.Queue[np.ndarray] | None = None
        self._asr_model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, audio_queue: queue.Queue[np.ndarray]) -> None:
        """Begin processing audio from *audio_queue* in a background thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._audio_queue = audio_queue
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"asr-{self.source_label}")
        self._thread.start()
        logger.info("ASR worker started for source=%s", self.source_label)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None
        logger.info("ASR worker stopped for source=%s", self.source_label)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_model(self):
        import nemo.collections.asr as asr

        logger.info("Loading ASR model: %s", self.model_name)
        self._asr_model = asr.models.ASRModel.from_pretrained(model_name=self.model_name)

        # Suppress NeMo/Lhotse logging AFTER model load — loggers are created lazily
        # during from_pretrained, so suppressing before has no effect on them.
        self._suppress_third_party_logging()

        # Fix config to eliminate warnings on every transcribe() call
        self._fix_model_config()

        logger.info("ASR model loaded")

    def _suppress_third_party_logging(self) -> None:
        from nemo.utils import logging as nemo_log

        # NeMo has its own Logger wrapper with its own handlers — clear them directly
        nemo_log.setLevel(logging.ERROR)
        if hasattr(nemo_log, "_logger"):
            nl = nemo_log._logger
            nl.handlers.clear()
            nl.addHandler(logging.NullHandler())
            nl.setLevel(logging.ERROR)
            nl.propagate = False

        # Suppress all noisy Python loggers (created lazily, so scan after model load)
        _noisy_prefixes = ("nemo", "lhotse", "nv_one_logger", "torch.distributed", "numexpr")
        for name in list(logging.root.manager.loggerDict):
            if any(name.lower().startswith(p) for p in _noisy_prefixes):
                lg = logging.getLogger(name)
                lg.handlers.clear()
                lg.addHandler(logging.NullHandler())
                lg.setLevel(logging.ERROR)
                lg.propagate = False
        # Lhotse uses the literal logger name "root"
        for name in ("root", "NeMo"):
            lg = logging.getLogger(name)
            lg.handlers.clear()
            lg.addHandler(logging.NullHandler())
            lg.setLevel(logging.ERROR)
            lg.propagate = False

    def _fix_model_config(self) -> None:
        from omegaconf import open_dict, OmegaConf

        cfg = self._asr_model.cfg
        cfg_yaml = OmegaConf.to_yaml(cfg)

        # Find which top-level config sections actually contain the noisy keys
        # so we know the real paths (they vary by model version)
        keys_found = [line.strip() for line in cfg_yaml.splitlines()
                      if "pretokenize" in line or "use_start_end_token" in line]
        if keys_found:
            logger.debug("Config keys to patch: %s", keys_found)

        with open_dict(cfg):
            for ds_key in ("train_ds", "validation_ds", "test_ds", "dataset", "model"):
                ds = getattr(cfg, ds_key, None)
                if ds is None:
                    continue
                if hasattr(ds, "pretokenize"):
                    ds.pretokenize = False
                if hasattr(ds, "use_start_end_token"):
                    del ds.use_start_end_token

    def _run(self) -> None:
        self._load_model()

        buffer = np.array([], dtype=np.float32)
        chunk_samples = int(self.chunk_seconds * self.sample_rate)
        overlap_samples = int(self.overlap_seconds * self.sample_rate)
        emitted_words: list[str] = []  # tail of previously emitted words for dedup
        capture_start = time.monotonic()

        while not self._stop_event.is_set():
            # Drain available audio from the queue
            got_audio = False
            try:
                while True:
                    chunk = self._audio_queue.get(timeout=0.1)
                    buffer = np.concatenate([buffer, chunk])
                    got_audio = True
            except queue.Empty:
                pass

            if not got_audio and len(buffer) < chunk_samples:
                continue

            if len(buffer) < chunk_samples:
                continue

            # We have enough audio — run inference
            audio_to_transcribe = buffer[:chunk_samples]
            # Keep overlap for next iteration
            buffer = buffer[chunk_samples - overlap_samples:]

            elapsed = time.monotonic() - capture_start
            new_words, new_segments = self._transcribe_chunk(audio_to_transcribe, elapsed, emitted_words)

            for w in new_words:
                self.store.add_word(w)
            for seg in new_segments:
                self.store.add_segment(seg)

        # Process any remaining audio in buffer on shutdown
        if len(buffer) >= self.sample_rate * 0.5:  # at least 0.5s
            elapsed = time.monotonic() - capture_start
            new_words, new_segments = self._transcribe_chunk(buffer, elapsed, emitted_words)
            for w in new_words:
                self.store.add_word(w)
            for seg in new_segments:
                self.store.add_segment(seg)

    def _transcribe_chunk(
        self,
        audio: np.ndarray,
        elapsed_offset: float,
        emitted_words: list[str],
    ) -> tuple[list[LiveWord], list[str]]:
        """Transcribe an audio array and return (new_words, new_segments).

        Deduplicates against the tail of *emitted_words* to avoid repeating
        words from the overlap region.
        """
        # Write to temporary WAV
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            sf.write(str(tmp_path), audio, self.sample_rate)
            outputs = self._asr_model.transcribe([str(tmp_path)], timestamps=True, verbose=False)
        finally:
            tmp_path.unlink(missing_ok=True)

        ts = outputs[0].timestamp
        raw_words: list[str] = [w["word"] for w in ts.get("word", [])]
        raw_word_ts = ts.get("word", [])
        raw_segments = ts.get("segment", [])

        # --- Deduplication via tail matching ---
        # Find how many words at the start of raw_words match the tail of
        # emitted_words (from the overlap region).
        skip = self._find_overlap(emitted_words, raw_words)

        new_live_words: list[LiveWord] = []
        for entry in raw_word_ts[skip:]:
            lw = LiveWord(
                timestamp=round(elapsed_offset + entry["start"], 3),
                word=entry["word"],
                source=self.source_label,
            )
            new_live_words.append(lw)
            emitted_words.append(entry["word"])

        # Keep emitted_words from growing unboundedly
        if len(emitted_words) > 200:
            del emitted_words[:len(emitted_words) - 100]

        new_segments: list[str] = []
        for seg in raw_segments[skip and 1 or 0:]:
            new_segments.append(seg["segment"])

        if new_live_words:
            logger.info(
                "Transcribed %d new word(s) (skipped %d overlap), source=%s",
                len(new_live_words), skip, self.source_label,
            )

        return new_live_words, new_segments

    @staticmethod
    def _find_overlap(emitted: list[str], new: list[str]) -> int:
        """Find the number of words at the start of *new* that match the
        tail of *emitted* (overlap region).

        Uses a simple greedy tail match: try matching the last N words of
        emitted against the first N words of new, for decreasing N.
        """
        if not emitted or not new:
            return 0

        max_check = min(len(emitted), len(new), 20)
        for length in range(max_check, 0, -1):
            tail = emitted[-length:]
            head = new[:length]
            if tail == head:
                return length
        return 0
