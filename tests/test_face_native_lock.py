"""Regression: every dlib entry point must share the same native lock."""
import ast
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from pathlib import Path
import threading
import time
import unittest

class FaceNativeLockTests(unittest.TestCase):
    def test_shared_lock_serializes_detector_and_encoder(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / "rtsp_yolo_backend.py").read_text(encoding='utf-8'))
        wrapper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "serialized_face_call")
        scope = {"wraps": wraps, "face_native_lock": threading.RLock()}
        exec(compile(ast.Module(body=[wrapper], type_ignores=[]), "<wrapper>", "exec"), scope)
        state = {"active": 0, "peak": 0}
        def native(value):
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
            try:
                time.sleep(.002)
                if value == 7:
                    raise ValueError("simulated native error")
                return value
            finally:
                state["active"] -= 1
        detector = scope["serialized_face_call"](native)
        encoder = scope["serialized_face_call"](native)
        def run(value):
            try:
                return (detector if value % 2 else encoder)(value)
            except ValueError:
                return None
        with ThreadPoolExecutor(max_workers=12) as pool:
            results = list(pool.map(run, range(60)))
        self.assertEqual(state["peak"], 1)
        self.assertEqual(results[59], 59)
        # Future call sites must use the guarded entry points too.
        unguarded = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Attribute)
                     and isinstance(n.func.value, ast.Name)
                     and n.func.value.id == "face_recognition"
                     and n.func.attr in {"face_locations", "face_encodings"}]
        self.assertEqual(unguarded, [])

if __name__ == "__main__":
    unittest.main()
