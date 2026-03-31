"""Thread-safe transcript accumulator for real-time transcription."""

from __future__ import annotations

import asyncio
import queue
import threading
from collections import deque

from models import LiveWord


class TranscriptStore:
    """Accumulates transcription results and serves them to API consumers.

    Thread-safe: the ASR worker thread writes via add_word/add_segment,
    while FastAPI handlers read via the query methods.
    """

    def __init__(self, max_words: int = 10_000, max_segments: int = 1_000) -> None:
        self._words: deque[LiveWord] = deque(maxlen=max_words)
        self._segments: deque[str] = deque(maxlen=max_segments)
        self._push_queue: queue.Queue[LiveWord] = queue.Queue()
        self._lock = threading.Lock()

        # WebSocket subscriber queues (one asyncio.Queue per connected client)
        self._ws_queues: list[asyncio.Queue[LiveWord]] = []
        self._ws_lock = threading.Lock()

    # ---- Write side (called from ASR worker thread) ----

    def add_word(self, word: LiveWord) -> None:
        with self._lock:
            self._words.append(word)
        self._push_queue.put_nowait(word)
        with self._ws_lock:
            for q in self._ws_queues:
                try:
                    q.put_nowait(word)
                except asyncio.QueueFull:
                    pass  # slow client, drop

    def add_segment(self, segment: str) -> None:
        with self._lock:
            self._segments.append(segment)

    # ---- Read side (called from API handlers) ----

    def last_word(self) -> LiveWord | None:
        with self._lock:
            return self._words[-1] if self._words else None

    def last_segment(self) -> str | None:
        with self._lock:
            return self._segments[-1] if self._segments else None

    def last_n_words(self, n: int) -> list[LiveWord]:
        with self._lock:
            items = list(self._words)
        return items[-n:] if n < len(items) else items

    def pop_word(self) -> LiveWord | None:
        try:
            return self._push_queue.get_nowait()
        except queue.Empty:
            return None

    def pop_words(self, n: int) -> list[LiveWord]:
        result: list[LiveWord] = []
        for _ in range(n):
            try:
                result.append(self._push_queue.get_nowait())
            except queue.Empty:
                break
        return result

    # ---- WebSocket subscription ----

    def subscribe_ws(self) -> asyncio.Queue[LiveWord]:
        q: asyncio.Queue[LiveWord] = asyncio.Queue(maxsize=500)
        with self._ws_lock:
            self._ws_queues.append(q)
        return q

    def unsubscribe_ws(self, q: asyncio.Queue[LiveWord]) -> None:
        with self._ws_lock:
            try:
                self._ws_queues.remove(q)
            except ValueError:
                pass

    # ---- Utility ----

    def word_count(self) -> int:
        with self._lock:
            return len(self._words)

    def clear(self) -> None:
        with self._lock:
            self._words.clear()
            self._segments.clear()
        # Drain the push queue
        while not self._push_queue.empty():
            try:
                self._push_queue.get_nowait()
            except queue.Empty:
                break
