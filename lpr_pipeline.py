"""Bounded, fair multi-frame review. No model imports or network side effects."""
from __future__ import annotations

import threading
import time
from collections import Counter

from lpr_utils import lpr_consensus, lpr_review_status, plate_read_kind


ACTIVE = {"queued", "reading", "pending"}


class LprPipeline:
    def __init__(self, create_event, update_event, *, clock=time.monotonic,
                 max_bytes=48 * 1024 * 1024, max_frames=3, max_attempts=5,
                 deadline=30.0, idle_timeout=3.0):
        self.create_event = create_event
        self.update_event = update_event
        self.clock = clock
        self.max_bytes = max_bytes
        self.max_frames = max_frames
        self.max_attempts = max_attempts
        self.deadline = deadline
        self.idle_timeout = idle_timeout
        self.lock = threading.RLock()
        self.tracks = {}
        self.generation = 0
        self.bytes = 0
        self.serial = 0
        self.first_reads = 0
        self.metrics = Counter()

    def _update(self, state, **changes):
        # Publish before changing memory: a failed DB write remains retryable.
        self.update_event(state["event"], changes)
        state.update(changes)

    def _finish(self, state, reason):
        plate = state.get("plate", "")
        kind = plate_read_kind(plate)
        if state.get("status") == "verified":
            status = "verified"
        elif kind == "complete" and state.get("confidence", 0) >= 60:
            status = "verified"
        elif kind == "partial":
            status = "partial"
        elif not plate:
            status = "unreadable"
        else:
            status = "uncertain"
        self._update(state, status=status, reason=reason)
        self.bytes -= sum(frame["bytes"] for frame in state["frames"])
        state["frames"].clear()
        self.metrics[reason] += 1

    def offer(self, track_id, frame_factory):
        """Register every observed vehicle; sample distinct frames at 250 ms.

        Frame factory encodes evidence only when needed. First observations
        create a durable event independently of OCR success or capacity.
        """
        with self.lock:
            now = self.clock()
            state = self.tracks.get(track_id)
            if state:
                state["last_seen"] = now
                if state["status"] not in ACTIVE or now - state["sampled"] < .25:
                    return
            frame = frame_factory()
            if frame is None:
                return
            if state is None:
                event = self.create_event(track_id, frame)
                state = dict(event=event, status="queued", reason="", plate="",
                             created=now, last_seen=now, sampled=now, last_run=0,
                             attempts=0, observations=[], frames=[], inflight=False)
                self.tracks[track_id] = state
                self.metrics["events"] += 1
            state["sampled"] = now
            self.serial += 1
            frame.update(sample_id=self.serial, bytes=sum(len(frame[k]) for k in ("vehicle", "scene", "crop")))
            if len(state["frames"]) >= self.max_frames:
                worst = min(state["frames"], key=lambda item: item["score"])
                if frame["score"] <= worst["score"]:
                    return
                state["frames"].remove(worst)
                self.bytes -= worst["bytes"]
            if self.bytes + frame["bytes"] > self.max_bytes:
                self.metrics["capacity_samples"] += 1
                if not state["frames"] and not state["inflight"]:
                    self._finish(state, "capacity")
                return
            state["frames"].append(frame)
            self.bytes += frame["bytes"]

    def expire(self):
        with self.lock:
            now = self.clock()
            for state in list(self.tracks.values()):
                if state["status"] not in ACTIVE:
                    continue
                if now - state["created"] >= self.deadline:
                    self._finish(state, "timeout")
                elif not state["inflight"] and not state["frames"] and now - state["last_seen"] >= self.idle_timeout:
                    self._finish(state, "no_more_frames")

    def next_job(self):
        with self.lock:
            self.expire()
            ready = [(tid, s) for tid, s in self.tracks.items()
                     if s["status"] in ACTIVE and s["frames"] and not s["inflight"]]
            fresh = [item for item in ready if item[1]["attempts"] == 0]
            retry = [item for item in ready if item[1]["attempts"] > 0]
            # At most three first reads before servicing the oldest retry.
            pool = retry if retry and (self.first_reads >= 3 or not fresh) else fresh
            if not pool:
                return None
            tid, state = min(pool, key=lambda item: (item[1]["last_run"], item[1]["created"]))
            self._update(state, status="reading", reason="")
            frame = max(state["frames"], key=lambda item: item["score"])
            state["frames"].remove(frame)
            self.bytes -= frame["bytes"]
            state["inflight"] = True
            state["attempts"] += 1
            state["last_run"] = self.clock()
            self.first_reads = self.first_reads + 1 if pool is fresh else 0
            self.metrics["ocr_started"] += 1
            return dict(track_id=tid, generation=self.generation,
                        attempt_index=state["attempts"] - 1, frame=frame)

    def complete(self, job, result=None, error=False):
        with self.lock:
            if job["generation"] != self.generation:
                return
            state = self.tracks.get(job["track_id"])
            if not state:
                return
            state["inflight"] = False
            if state["status"] not in ACTIVE:
                return  # A late OCR result cannot revive an expired event.
            self.metrics["ocr_finished"] += 1
            if error:
                self._finish(state, "ocr_error")
                return
            if result and plate_read_kind(result["plate"]) != "invalid":
                observation = dict(result, sample_id=job["frame"]["sample_id"], frame=job["frame"])
                state["observations"].append(observation)
                consensus = lpr_consensus(state["observations"])
                status = lpr_review_status(state["observations"])
                winner = max((o for o in state["observations"]
                              if o["plate"] == consensus["plate"]), key=lambda o: o["conf"])
                self._update(state, status=status, plate=consensus["plate"],
                             confidence=consensus["confidence"],
                             initial_plate=state["observations"][0]["plate"],
                             sample_count=consensus["sample_count"],
                             plate_crop=winner.get("plate_crop"),
                             winning_frame=winner.get("frame"), reason="")
            else:
                self._update(state, status="pending", reason="")
            if state["status"] not in ACTIVE:
                self.bytes -= sum(f["bytes"] for f in state["frames"])
                state["frames"].clear()
            elif state["attempts"] >= self.max_attempts:
                self._finish(state, "attempt_limit")
            self.expire()

    def reset(self, reason="interrupted"):
        with self.lock:
            for state in self.tracks.values():
                if state["status"] in ACTIVE:
                    self._finish(state, reason)
            self.generation += 1
            self.tracks.clear()
            self.bytes = 0

    def forget(self, track_id):
        with self.lock:
            state = self.tracks.get(track_id)
            if state and state["status"] not in ACTIVE and not state["inflight"]:
                self.tracks.pop(track_id, None)

    def snapshot(self):
        with self.lock:
            return dict(self.metrics, waiting=sum(s["status"] in ACTIVE for s in self.tracks.values()),
                        buffered_bytes=self.bytes,
                        oldest_wait_seconds=round(max((self.clock() - s["created"] for s in self.tracks.values()
                                                      if s["status"] in ACTIVE), default=0), 1))
