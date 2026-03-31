"""Audio capture from microphone, desktop (WASAPI loopback), or network stream."""

from __future__ import annotations

import logging
import queue
import subprocess
import sys
import threading
from enum import Enum

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class AudioSource(str, Enum):
    MIC = "mic"
    DESKTOP = "desktop"
    BOTH = "both"
    NETWORK = "network"


class MixMode(str, Enum):
    SEPARATE = "separate"
    MIXED = "mixed"


def _find_wasapi_loopback_device() -> int | None:
    """Find the default WASAPI loopback device index on Windows."""
    try:
        hostapis = sd.query_hostapis()
    except Exception:
        return None

    wasapi_api = None
    for api in hostapis:
        if "wasapi" in api["name"].lower():
            wasapi_api = api
            break

    if wasapi_api is None:
        return None

    # The default output device's loopback is what we want.
    # On WASAPI, loopback devices have "Loopback" in the name or are
    # identified by the is_loopback flag when queried with wasapi settings.
    devices = sd.query_devices()
    default_output = wasapi_api.get("default_output_device", -1)

    # Try to find a loopback device matching the default output
    for i, dev in enumerate(devices):
        name = dev["name"].lower()
        if (
            dev["hostapi"] == hostapis.index(wasapi_api)
            and dev["max_input_channels"] > 0
            and ("loopback" in name or i == default_output)
        ):
            return i

    # Fallback: any WASAPI input device (some configs expose loopback as input)
    for i, dev in enumerate(devices):
        if (
            dev["hostapi"] == hostapis.index(wasapi_api)
            and dev["max_input_channels"] > 0
        ):
            return i

    return None


class AudioCapture:
    """Captures audio from mic and/or desktop and pushes chunks to queues.

    Parameters
    ----------
    source : AudioSource
        Which audio source(s) to capture.
    mix_mode : MixMode
        When source is BOTH: "separate" gives two independent queues,
        "mixed" combines them into one queue.
    sample_rate : int
        Target sample rate (NeMo expects 16000 Hz).
    chunk_duration : float
        Duration of each audio chunk in seconds.
    device_index : int | None
        Override the audio device index (applies to mic for MIC source,
        loopback for DESKTOP source).
    """

    def __init__(
        self,
        source: AudioSource = AudioSource.MIC,
        mix_mode: MixMode = MixMode.SEPARATE,
        sample_rate: int = 16_000,
        chunk_duration: float = 0.5,
        device_index: int | None = None,
        stream_url: str | None = None,
    ) -> None:
        self.source = source
        self.mix_mode = mix_mode
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.device_index = device_index
        self.stream_url = stream_url

        self._streams: list[sd.InputStream] = []
        self._running = False
        self._stop_event = threading.Event()
        self._ffmpeg_proc: subprocess.Popen | None = None
        self._network_thread: threading.Thread | None = None

        # Output queues — consumers read from these
        self.mic_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=60)
        self.desktop_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=60)
        self.mixed_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=60)
        self.network_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=60)

    @property
    def is_running(self) -> bool:
        return self._running

    def get_queues(self) -> dict[str, queue.Queue[np.ndarray]]:
        """Return a mapping of source label -> audio queue for the ASR workers."""
        if self.source == AudioSource.MIC:
            return {"mic": self.mic_queue}
        elif self.source == AudioSource.DESKTOP:
            return {"desktop": self.desktop_queue}
        elif self.source == AudioSource.NETWORK:
            return {"network": self.network_queue}
        elif self.mix_mode == MixMode.MIXED:
            return {"mixed": self.mixed_queue}
        else:
            return {"mic": self.mic_queue, "desktop": self.desktop_queue}

    def _make_callback(self, target_queue: queue.Queue[np.ndarray], native_rate: int):
        """Create a sounddevice callback that resamples and enqueues audio."""

        need_resample = native_rate != self.sample_rate

        def callback(indata: np.ndarray, frames: int, time_info, status) -> None:
            if status:
                logger.warning("Audio callback status: %s", status)

            # Convert to mono float32 if needed
            audio = indata[:, 0].copy() if indata.ndim > 1 else indata.copy().flatten()

            # Resample if native rate differs from target
            if need_resample:
                from scipy.signal import resample_poly
                from math import gcd
                g = gcd(self.sample_rate, native_rate)
                audio = resample_poly(audio, self.sample_rate // g, native_rate // g).astype(np.float32)

            try:
                target_queue.put_nowait(audio)
            except queue.Full:
                # Drop oldest chunk to make room
                try:
                    target_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    target_queue.put_nowait(audio)
                except queue.Full:
                    pass

        return callback

    def _make_mixed_callback(self, native_rate: int):
        """Callback for mixed mode — sends to both individual and mixed queues."""

        need_resample = native_rate != self.sample_rate

        def callback(indata: np.ndarray, frames: int, time_info, status) -> None:
            if status:
                logger.warning("Audio callback status: %s", status)

            audio = indata[:, 0].copy() if indata.ndim > 1 else indata.copy().flatten()

            if need_resample:
                from scipy.signal import resample_poly
                from math import gcd
                g = gcd(self.sample_rate, native_rate)
                audio = resample_poly(audio, self.sample_rate // g, native_rate // g).astype(np.float32)

            try:
                self.mixed_queue.put_nowait(audio)
            except queue.Full:
                try:
                    self.mixed_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.mixed_queue.put_nowait(audio)
                except queue.Full:
                    pass

        return callback

    def _build_ffmpeg_cmd(self) -> list[str]:
        cmd = [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "warning",
        ]
        url = self.stream_url
        if url.startswith(("rtp://", "udp://")):
            cmd += ["-protocol_whitelist", "file,udp,rtp,pipe,crypto"]
        cmd += [
            "-i", url,
            "-vn",                        # drop video tracks
            "-acodec", "pcm_s16le",
            "-ar", str(self.sample_rate),
            "-ac", "1",
            "-f", "s16le",
            "pipe:1",
        ]
        return cmd

    def _run_network_reader(self) -> None:
        chunk_samples = int(self.sample_rate * self.chunk_duration)
        bytes_per_chunk = chunk_samples * 2  # int16 = 2 bytes per sample

        cmd = self._build_ffmpeg_cmd()
        logger.info("Network stream: url=%s", self.stream_url)
        logger.debug("ffmpeg command: %s", " ".join(cmd))

        try:
            self._ffmpeg_proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=None, bufsize=0,
            )
        except FileNotFoundError:
            logger.error("ffmpeg not found on PATH — cannot start network stream")
            return

        proc = self._ffmpeg_proc
        try:
            while not self._stop_event.is_set():
                raw = proc.stdout.read(bytes_per_chunk)
                if not raw:
                    if not self._stop_event.is_set():
                        logger.warning("Network stream ended (ffmpeg exited)")
                    break
                if len(raw) < bytes_per_chunk:
                    continue

                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                try:
                    self.network_queue.put_nowait(audio)
                except queue.Full:
                    try:
                        self.network_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self.network_queue.put_nowait(audio)
                    except queue.Full:
                        pass
        finally:
            proc.stdout.close()
            proc.wait(timeout=3)
            logger.info("Network reader thread exiting")

    def _open_mic_stream(self, target_queue: queue.Queue[np.ndarray]) -> sd.InputStream:
        device = self.device_index if self.source == AudioSource.MIC else None
        device_info = sd.query_devices(device or sd.default.device[0], "input")
        native_rate = int(device_info["default_samplerate"])
        blocksize = int(native_rate * self.chunk_duration)

        stream = sd.InputStream(
            device=device,
            channels=1,
            samplerate=native_rate,
            dtype="float32",
            blocksize=blocksize,
            callback=self._make_callback(target_queue, native_rate),
        )
        logger.info("Mic stream: device=%s, native_rate=%d, blocksize=%d", device_info["name"], native_rate, blocksize)
        return stream

    def _open_desktop_stream(self, target_queue: queue.Queue[np.ndarray]) -> sd.InputStream:
        if sys.platform != "win32":
            raise RuntimeError("Desktop audio capture (WASAPI loopback) is only supported on Windows")

        device = self.device_index if self.source == AudioSource.DESKTOP else None
        if device is None:
            device = _find_wasapi_loopback_device()
        if device is None:
            raise RuntimeError(
                "Could not find a WASAPI loopback device. "
                "Use --device-index to specify one manually. "
                "List devices with: python -c \"import sounddevice; print(sounddevice.query_devices())\""
            )

        device_info = sd.query_devices(device, "input")
        native_rate = int(device_info["default_samplerate"])
        blocksize = int(native_rate * self.chunk_duration)

        stream = sd.InputStream(
            device=device,
            channels=1,
            samplerate=native_rate,
            dtype="float32",
            blocksize=blocksize,
            callback=self._make_callback(target_queue, native_rate),
            extra_settings=sd.WasapiSettings(exclusive=False),
        )
        logger.info("Desktop stream: device=%s, native_rate=%d, blocksize=%d", device_info["name"], native_rate, blocksize)
        return stream

    def start(self) -> None:
        if self._running:
            return

        if self.source in (AudioSource.MIC, AudioSource.BOTH):
            if self.source == AudioSource.BOTH and self.mix_mode == MixMode.MIXED:
                stream = self._open_mic_stream(self.mixed_queue)
            else:
                stream = self._open_mic_stream(self.mic_queue)
            self._streams.append(stream)

        if self.source in (AudioSource.DESKTOP, AudioSource.BOTH):
            if self.source == AudioSource.BOTH and self.mix_mode == MixMode.MIXED:
                stream = self._open_desktop_stream(self.mixed_queue)
            else:
                stream = self._open_desktop_stream(self.desktop_queue)
            self._streams.append(stream)

        if self.source == AudioSource.NETWORK:
            if not self.stream_url:
                raise ValueError("stream_url is required when source=NETWORK")
            self._stop_event.clear()
            self._network_thread = threading.Thread(
                target=self._run_network_reader,
                daemon=True,
                name="network-stream-reader",
            )
            self._network_thread.start()

        for stream in self._streams:
            stream.start()
        self._running = True
        logger.info("Audio capture started: source=%s, mix_mode=%s", self.source.value, self.mix_mode.value)

    def stop(self) -> None:
        if not self._running:
            return

        for stream in self._streams:
            stream.stop()
            stream.close()
        self._streams.clear()

        if self._network_thread is not None:
            self._stop_event.set()
            if self._ffmpeg_proc is not None:
                self._ffmpeg_proc.terminate()
            self._network_thread.join(timeout=5)
            self._network_thread = None
            self._ffmpeg_proc = None

        self._running = False
        logger.info("Audio capture stopped")
