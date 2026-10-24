import os
import unittest
from pathlib import Path
from unittest.mock import patch
import bootstrap
from optional_face import load_face_backend

class InstallProfilesTests(unittest.TestCase):
    def test_vehicle_does_not_import_face_package(self):
        import builtins
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name in {'face_recognition', 'dlib'}:
                raise AssertionError('Vehicle profile imported native face dependency')
            return original(name, *args, **kwargs)
        with patch.dict(os.environ, {'CCTV_PROFILE':'vehicle'}), patch('builtins.__import__', guarded):
            backend, available=load_face_backend()
            self.assertFalse(available)
            self.assertEqual(backend.face_locations(None), [])
            self.assertEqual(backend.face_encodings(None), [])

    def test_mac_full_keeps_face_backend(self):
        import sys
        sentinel=object()
        with patch.dict(os.environ, {'CCTV_PROFILE':'full'}), patch.dict(sys.modules, {'face_recognition':sentinel}):
            backend, available=load_face_backend()
            self.assertIs(backend,sentinel)
            self.assertTrue(available)

    def test_requirement_profiles(self):
        root=Path(__file__).resolve().parents[1]
        self.assertNotIn('face-recognition',(root/'requirements-core.txt').read_text())
        self.assertIn('face-recognition',(root/'requirements.txt').read_text())
        self.assertIn('-r requirements-core.txt',(root/'requirements-windows.txt').read_text())

    def test_profile_changes_dependency_stamp(self):
        with patch.dict(os.environ, {'CCTV_PROFILE':'full'}):
            full=bootstrap.requirements_digest()
        with patch.dict(os.environ, {'CCTV_PROFILE':'vehicle'}):
            self.assertNotEqual(full,bootstrap.requirements_digest())

if __name__=='__main__':unittest.main()
