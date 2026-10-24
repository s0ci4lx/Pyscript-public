"""Small helpers for keeping video delivery responsive under AI load."""

from __future__ import annotations

import threading

import cv2


class JpegFrameCache:
    """Encode each frame object once, then share the JPEG between viewers."""

    def __init__(self, quality: int = 80):
        self.quality = int(quality)
        self._lock = threading.Lock()
        self._frame_ref = None
        self._payload = None

    def encode(self, frame, encoder=None):
        if frame is None:
            return None

        encode_fn = encoder or cv2.imencode
        with self._lock:
            if frame is self._frame_ref and self._payload is not None:
                return self._payload

            ok, buffer = encode_fn(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, self.quality],
            )
            if not ok:
                return None

            self._frame_ref = frame
            self._payload = buffer.tobytes()
            return self._payload


def video_catchup_frames(processing_seconds: float, native_fps: float, max_seconds: float = 3.0) -> int:
    """Frames to drop after slow processing so file playback stays near real time."""
    processing_seconds = max(0.0, float(processing_seconds or 0.0))
    native_fps = max(0.0, float(native_fps or 0.0))
    if native_fps <= 0.0:
        return 0

    elapsed_frames = int(processing_seconds * native_fps)
    maximum = max(0, int(native_fps * max(0.0, float(max_seconds))))
    return min(maximum, max(0, elapsed_frames - 1))


class LatestFrameCapture:
    """Own a live decoder on one thread; bound latency to one pending frame."""

    def __init__(self, factory, timeout=8.0):
        import time
        self._clock = time.monotonic
        self._factory = factory
        self._timeout = timeout
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._alive = True
        self._frame = None
        self._captured_at = 0.0
        self.received = self.skipped = self.delivered = 0
        self.last_age_ms = 0.0
        self._thread = threading.Thread(target=self._produce, daemon=True)
        self._thread.start()
        self._ready.wait(timeout)

    def _produce(self):
        capture = None
        try:
            capture = self._factory()
            self._ready.set()
            while not self._stop.is_set() and capture.isOpened():
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                with self._condition:
                    self.received += 1
                    if self._frame is not None:
                        self.skipped += 1
                    self._frame = frame
                    self._captured_at = self._clock()
                    self._condition.notify_all()
        finally:
            if capture is not None:
                capture.release()
            with self._condition:
                self._alive = False
                self._condition.notify_all()
            self._ready.set()

    def isOpened(self):
        with self._condition:
            return not self._stop.is_set() and (self._alive or self._frame is not None)

    def read(self):
        deadline = self._clock() + self._timeout
        with self._condition:
            while self._frame is None and self._alive and not self._stop.is_set():
                remaining = deadline - self._clock()
                if remaining <= 0:
                    return False, None
                self._condition.wait(remaining)
            if self._frame is None or self._stop.is_set():
                return False, None
            frame, self._frame = self._frame, None
            self.delivered += 1
            self.last_age_ms = (self._clock() - self._captured_at) * 1000
            return True, frame

    def release(self):
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        # Only the producer may touch/release the native decoder.
        self._thread.join(timeout=self._timeout + 1)

    def status(self):
        with self._condition:
            return dict(received=self.received, skipped=self.skipped,
                        delivered=self.delivered, last_age_ms=round(self.last_age_ms, 1))
