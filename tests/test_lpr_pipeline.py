import time
import unittest

from lpr_pipeline import LprPipeline


class LprPipelineTests(unittest.TestCase):
    def setUp(self):
        self.created_events = []
        self.updated_events = []
        self.now = 1000.0

    def clock(self):
        return self.now

    def create_event(self, track_id, frame):
        event = {"id": len(self.created_events) + 1, "track_id": track_id, "frame": frame}
        self.created_events.append(event)
        return event

    def update_event(self, event, changes):
        self.updated_events.append((event["id"], dict(changes)))

    def sample_frame(self, score=100.0, size=1024):
        return {
            "vehicle": b"v" * size,
            "scene": b"s" * size,
            "crop": b"c" * size,
            "score": score,
        }

    def test_offer_creates_event_and_buffers_frames(self):
        pipeline = LprPipeline(self.create_event, self.update_event, clock=self.clock)
        pipeline.offer(1, lambda: self.sample_frame(score=50.0))

        self.assertEqual(len(self.created_events), 1)
        self.assertEqual(self.created_events[0]["track_id"], 1)

        # Within 250ms, offer does not generate a new frame
        called = [False]
        def should_not_run():
            called[0] = True
            return self.sample_frame()

        pipeline.offer(1, should_not_run)
        self.assertFalse(called[0])

    def test_next_job_serves_jobs_and_handles_completion(self):
        pipeline = LprPipeline(self.create_event, self.update_event, clock=self.clock)
        pipeline.offer(10, lambda: self.sample_frame(score=60.0))

        job = pipeline.next_job()
        self.assertIsNotNone(job)
        self.assertEqual(job["track_id"], 10)
        self.assertEqual(job["attempt_index"], 0)

        # Complete with verified plate
        pipeline.complete(job, result={"plate": "สง 2277", "conf": 92})

        track_state = pipeline.tracks[10]
        self.assertEqual(track_state["status"], "verified")
        self.assertEqual(track_state["plate"], "สง 2277")

    def test_partial_plate_not_verified_and_becomes_partial_on_exit(self):
        pipeline = LprPipeline(self.create_event, self.update_event, clock=self.clock, idle_timeout=2.0)
        pipeline.offer(20, lambda: self.sample_frame(score=80.0))

        job = pipeline.next_job()
        pipeline.complete(job, result={"plate": "5762", "conf": 98})

        # While active, partial plate must remain pending (not verified!)
        track_state = pipeline.tracks[20]
        self.assertEqual(track_state["status"], "pending")
        self.assertEqual(track_state["plate"], "5762")

        # Vehicle leaves, idle timeout expires
        self.now += 3.0
        pipeline.expire()

        self.assertEqual(track_state["status"], "partial")
        self.assertEqual(track_state["reason"], "no_more_frames")

    def test_unreadable_when_no_plate_found_and_car_leaves(self):
        pipeline = LprPipeline(self.create_event, self.update_event, clock=self.clock, idle_timeout=2.0)
        pipeline.offer(30, lambda: self.sample_frame(score=70.0))

        job = pipeline.next_job()
        # OCR found nothing or invalid string
        pipeline.complete(job, result=None)

        track_state = pipeline.tracks[30]
        self.assertEqual(track_state["status"], "pending")

        # Vehicle leaves, idle timeout expires
        self.now += 3.0
        pipeline.expire()

        self.assertEqual(track_state["status"], "unreadable")
        self.assertEqual(track_state["reason"], "no_more_frames")

    def test_complete_plate_verified_on_exit_with_good_confidence(self):
        pipeline = LprPipeline(self.create_event, self.update_event, clock=self.clock, idle_timeout=2.0)
        pipeline.offer(35, lambda: self.sample_frame(score=75.0))

        job = pipeline.next_job()
        pipeline.complete(job, result={"plate": "งค 8754", "conf": 68})

        track_state = pipeline.tracks[35]
        # While active, 68% single sample was pending
        self.assertEqual(track_state["status"], "pending")

        # Vehicle leaves, idle timeout expires
        self.now += 3.0
        pipeline.expire()

        # Should be verified, NOT demoted to uncertain!
        self.assertEqual(track_state["status"], "verified")
        self.assertEqual(track_state["plate"], "งค 8754")

    def test_reset_marks_active_tracks_interrupted_and_increments_generation(self):
        pipeline = LprPipeline(self.create_event, self.update_event, clock=self.clock)
        pipeline.offer(40, lambda: self.sample_frame(score=70.0))

        old_gen = pipeline.generation
        pipeline.reset("seek")

        self.assertEqual(pipeline.generation, old_gen + 1)
        self.assertEqual(len(pipeline.tracks), 0)
        self.assertEqual(pipeline.bytes, 0)


if __name__ == "__main__":
    unittest.main()
