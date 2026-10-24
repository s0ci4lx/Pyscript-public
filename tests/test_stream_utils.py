import numpy as np

from stream_utils import JpegFrameCache, video_catchup_frames


class _Buffer:
    def __init__(self, payload):
        self.payload = payload

    def tobytes(self):
        return self.payload


def test_jpeg_cache_encodes_same_frame_only_once():
    calls = []

    def fake_encoder(extension, frame, options):
        calls.append((extension, frame, options))
        return True, _Buffer(b"jpeg-data")

    cache = JpegFrameCache(quality=77)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    assert cache.encode(frame, encoder=fake_encoder) == b"jpeg-data"
    assert cache.encode(frame, encoder=fake_encoder) == b"jpeg-data"
    assert len(calls) == 1
    assert calls[0][2][-1] == 77


def test_jpeg_cache_reencodes_a_new_frame_object():
    calls = []

    def fake_encoder(extension, frame, options):
        calls.append(frame)
        return True, _Buffer(bytes([len(calls)]))

    cache = JpegFrameCache()
    first = np.zeros((2, 2, 3), dtype=np.uint8)
    second = first.copy()

    assert cache.encode(first, encoder=fake_encoder) == b"\x01"
    assert cache.encode(second, encoder=fake_encoder) == b"\x02"
    assert len(calls) == 2


def test_video_catchup_drops_only_frames_that_are_late():
    assert video_catchup_frames(0.02, 25) == 0
    assert video_catchup_frames(0.04, 25) == 0
    assert video_catchup_frames(0.20, 25) == 4
    assert video_catchup_frames(10.0, 25) == 75
    assert video_catchup_frames(1.0, 0) == 0
