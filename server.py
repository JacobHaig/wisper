"""FastAPI server exposing real-time transcription via REST and WebSocket."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect

from models import LiveWord

if TYPE_CHECKING:
    from transcript_store import TranscriptStore

# ---------------------------------------------------------------------------
# The app is created here but the stores/state are injected at startup by
# serve.py via the lifespan context manager.
# ---------------------------------------------------------------------------

app = FastAPI(title="Wisper Real-Time Transcription")

# These are populated by the lifespan in serve.py before any request arrives.
stores: dict[str, "TranscriptStore"] = {}  # source_label -> store
_source_labels: list[str] = []
_status_info: dict = {}


def _resolve_store(source: str | None) -> "TranscriptStore":
    """Return the TranscriptStore for the given source label."""
    if not stores:
        raise RuntimeError("No transcript stores configured")
    if source is None:
        # Default to the first (or only) store
        return next(iter(stores.values()))
    if source not in stores:
        available = ", ".join(stores.keys())
        raise ValueError(f"Unknown source '{source}'. Available: {available}")
    return stores[source]


def _live_word_dict(w: LiveWord) -> dict:
    return asdict(w)


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/api/status")
def status():
    return {
        **_status_info,
        "sources": _source_labels,
        "word_counts": {label: store.word_count() for label, store in stores.items()},
    }


@app.get("/api/last-word")
def last_word(source: str | None = Query(default=None)):
    store = _resolve_store(source)
    w = store.last_word()
    return _live_word_dict(w) if w else None


@app.get("/api/last-segment")
def last_segment(source: str | None = Query(default=None)):
    store = _resolve_store(source)
    seg = store.last_segment()
    return {"segment": seg} if seg else None


@app.get("/api/last-words")
def last_words(n: int = Query(default=100, ge=1, le=10000), source: str | None = Query(default=None)):
    store = _resolve_store(source)
    words = store.last_n_words(n)
    return {"words": [_live_word_dict(w) for w in words]}


@app.get("/api/pop-word")
def pop_word(source: str | None = Query(default=None)):
    store = _resolve_store(source)
    w = store.pop_word()
    return _live_word_dict(w) if w else None


@app.get("/api/pop-words")
def pop_words(n: int = Query(default=10, ge=1, le=1000), source: str | None = Query(default=None)):
    store = _resolve_store(source)
    words = store.pop_words(n)
    return {"words": [_live_word_dict(w) for w in words]}


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket, source: str | None = Query(default=None)):
    store = _resolve_store(source)
    await websocket.accept()

    ws_queue = store.subscribe_ws()
    try:
        while True:
            word: LiveWord = await ws_queue.get()
            await websocket.send_json(_live_word_dict(word))
    except WebSocketDisconnect:
        pass
    finally:
        store.unsubscribe_ws(ws_queue)
