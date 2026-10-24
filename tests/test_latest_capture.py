import threading
import time
import unittest
from stream_utils import LatestFrameCapture

class LatestCaptureTests(unittest.TestCase):
    def test_latest_frame_and_decoder_ownership(self):
        owners = []
        finished = threading.Event()
        class Fake:
            def __init__(self): self.i=0; owners.append(threading.get_ident())
            def isOpened(self): return self.i<40
            def read(self):
                owners.append(threading.get_ident())
                time.sleep(.002); self.i+=1
                return True,self.i
            def release(self): owners.append(threading.get_ident()); finished.set()
        capture=LatestFrameCapture(Fake)
        self.assertTrue(finished.wait(3))
        ok,frame=capture.read()
        self.assertTrue(ok)
        self.assertEqual(frame,40)
        self.assertEqual(capture.status()['skipped'],39)
        self.assertFalse(capture.isOpened())
        capture.release()
        self.assertEqual(len(set(owners)),1)

    def test_release_wakes_waiting_consumer(self):
        stop=threading.Event()
        class Fake:
            def isOpened(self): return not stop.is_set()
            def read(self): stop.wait(.08); return False,None
            def release(self): stop.set()
        capture=LatestFrameCapture(Fake)
        results=[]
        consumer=threading.Thread(target=lambda:results.append(capture.read()))
        consumer.start();capture.release();consumer.join(1)
        self.assertEqual(results,[(False,None)])

if __name__=='__main__':unittest.main()
