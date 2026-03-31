"""CLI entry point for the wisper real-time transcription service.

Usage:
    uv run serve.py                                          # mic only, default settings
    uv run serve.py --source desktop                         # desktop audio (WASAPI loopback)
    uv run serve.py --source both                            # mic + desktop
    uv run serve.py --source both --mix mixed                # mic + desktop mixed into one stream
    uv run serve.py --source network --stream-url rtp://0.0.0.0:5004   # RTP stream
    uv run serve.py --stream-url rtsp://192.168.1.10/live    # RTSP (auto-sets --source=network)
    uv run serve.py --port 9000                              # custom port
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from contextlib import asynccontextmanager

# Suppress tqdm progress bars (NeMo prints "Transcribing: 1it [...]" per chunk)
os.environ.setdefault("TQDM_DISABLE", "1")

import uvicorn
from fastapi import FastAPI

from audio_capture import AudioCapture, AudioSource, MixMode
from realtime_asr import ChunkedParakeetWorker
from transcript_store import TranscriptStore

logger = logging.getLogger("wisper")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="wisper — real-time audio transcription service",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["mic", "desktop", "both", "network"],
        default="mic",
        help="Audio source (default: mic)",
    )
    parser.add_argument(
        "--stream-url",
        type=str,
        default=None,
        metavar="URL",
        help="Network audio stream URL, e.g. rtp://0.0.0.0:5004 or rtsp://host/path. Implies --source=network.",
    )
    parser.add_argument(
        "--mix",
        type=str,
        choices=["separate", "mixed"],
        default="separate",
        help="When --source=both: keep separate transcripts or mix into one (default: separate)",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=3.0,
        help="Seconds of audio to accumulate before ASR inference (default: 3.0)",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=None,
        help="Override audio device index (use 'python -c \"import sounddevice; print(sounddevice.query_devices())\"' to list)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="nvidia/parakeet-tdt-0.6b-v3",
        help="ASR model name (default: nvidia/parakeet-tdt-0.6b-v3)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # NeMo/Lhotse use their own handlers with custom formatters that ignore
    # logger.setLevel(). Filter by source file path instead — reliable regardless
    # of how NeMo routes its messages internally.
    _BLOCKED_PATHS = ("nemo", "lhotse", "nv_one_logger", "numexpr", "torch" + "/distributed")

    class _BlockThirdPartyFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            path = record.pathname.lower().replace("\\", "/")
            return not any(p in path for p in _BLOCKED_PATHS)

    for _handler in logging.root.handlers:
        _handler.addFilter(_BlockThirdPartyFilter())


    # --stream-url implies --source=network
    if args.stream_url and args.source == "mic":
        args.source = "network"
    if args.source == "network" and not args.stream_url:
        print("Error: --source=network requires --stream-url", file=sys.stderr)
        sys.exit(1)

    source = AudioSource(args.source)
    mix_mode = MixMode(args.mix)

    if args.chunk_seconds < 1.0:
        print("Error: --chunk-seconds must be >= 1.0", file=sys.stderr)
        sys.exit(1)

    # --- Build components (not started yet — lifespan will start them) ---
    capture = AudioCapture(
        source=source,
        mix_mode=mix_mode,
        device_index=args.device_index,
        stream_url=args.stream_url,
    )

    audio_queues = capture.get_queues()

    # Create a TranscriptStore and ASR worker per audio source queue
    stores: dict[str, TranscriptStore] = {}
    workers: list[ChunkedParakeetWorker] = []

    for label, q in audio_queues.items():
        store = TranscriptStore()
        stores[label] = store
        worker = ChunkedParakeetWorker(
            store=store,
            source_label=label,
            model_name=args.model,
            chunk_seconds=args.chunk_seconds,
        )
        workers.append((worker, q))

    # --- Lifespan: start/stop capture + ASR workers with the server ---
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Import and configure the server module's shared state
        import server

        server.stores = stores
        server._source_labels = list(stores.keys())
        server._status_info = {
            "running": True,
            "source": source.value,
            "mix_mode": mix_mode.value,
            "model": args.model,
            "chunk_seconds": args.chunk_seconds,
        }

        # Start audio capture
        logger.info("Starting audio capture: source=%s, mix=%s", source.value, mix_mode.value)
        capture.start()

        # Start ASR workers
        for worker, q in workers:
            worker.start(q)

        logger.info("Service ready at http://%s:%d", args.host, args.port)
        logger.info("  Sources: %s", ", ".join(stores.keys()))
        logger.info("  Endpoints: GET /api/status, /api/last-word, /api/last-segment, /api/last-words, /api/pop-word, /api/pop-words")
        logger.info("  WebSocket: ws://%s:%d/ws/stream", args.host, args.port)

        yield

        # Shutdown
        logger.info("Shutting down...")
        for worker, _ in workers:
            worker.stop()
        capture.stop()
        server._status_info["running"] = False

    # --- Import the FastAPI app and attach lifespan ---
    from server import app

    app.router.lifespan_context = lifespan

    # --- Run ---
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
