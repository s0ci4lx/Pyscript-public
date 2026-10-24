import numpy as np
import pytest
from rtsp_yolo_backend import CameraWorker, check_bbox_polygon_intersection


def test_assign_fallback_track_ids():
    worker = CameraWorker("test_cam", "Test Cam", "0")

    # Frame 1: Two boxes detected without tracker IDs
    boxes_f1 = np.array([[100, 100, 200, 200], [400, 400, 500, 500]], dtype=float)
    cls_ids_f1 = np.array([2, 7], dtype=int) # car, truck

    ids_f1 = worker._assign_fallback_track_ids(boxes_f1, cls_ids_f1)
    assert len(ids_f1) == 2
    assert ids_f1[0] != ids_f1[1]

    # Frame 2: Boxes slightly moved (high IoU)
    boxes_f2 = np.array([[105, 105, 205, 205], [402, 402, 502, 502]], dtype=float)
    cls_ids_f2 = np.array([2, 7], dtype=int)

    ids_f2 = worker._assign_fallback_track_ids(boxes_f2, cls_ids_f2)
    assert len(ids_f2) == 2
    # Should maintain the same track IDs across frames
    assert ids_f2[0] == ids_f1[0]
    assert ids_f2[1] == ids_f1[1]

    # Frame 3: Brand new box at different location
    boxes_f3 = np.array([[700, 700, 800, 800]], dtype=float)
    cls_ids_f3 = np.array([3], dtype=int) # motorcycle

    ids_f3 = worker._assign_fallback_track_ids(boxes_f3, cls_ids_f3)
    assert len(ids_f3) == 1
    assert ids_f3[0] != ids_f1[0]
    assert ids_f3[0] != ids_f1[1]


def test_check_bbox_polygon_intersection():
    pts_arr = np.array([[200, 200], [600, 200], [600, 600], [200, 600]], dtype=np.int32)

    # Box clearly inside
    assert check_bbox_polygon_intersection(300, 300, 400, 400, pts_arr) is True

    # Box overlapping boundary
    assert check_bbox_polygon_intersection(150, 150, 250, 250, pts_arr) is True

    # Box completely outside
    assert check_bbox_polygon_intersection(10, 10, 50, 50, pts_arr) is False
