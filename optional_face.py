"""Keep vehicle-only Windows installs independent of native dlib."""
import os

class UnavailableFaces:
    available = False
    def face_locations(self, *args, **kwargs):
        return []
    def face_encodings(self, *args, **kwargs):
        return []
    def load_image_file(self, *args, **kwargs):
        raise RuntimeError("Face recognition is unavailable in the vehicle profile")

def load_face_backend():
    profile = os.environ.get("CCTV_PROFILE", "vehicle" if os.name == "nt" else "full")
    if profile == "vehicle":
        return UnavailableFaces(), False
    # Mac/full installs retain the original dependency and behavior.
    import face_recognition
    return face_recognition, True
