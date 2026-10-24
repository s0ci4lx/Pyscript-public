import numpy as np

from lpr_utils import (
    expanded_plate_bbox,
    lpr_consensus,
    lpr_frame_quality_score,
    lpr_review_status,
    lpr_upscale_factor,
    normalize_thai_plate_digits,
    plate_read_kind,
    prepare_lpr_ocr_images,
)


def test_lpr_consensus_replaces_a_low_confidence_first_read():
    observations = [
        {"plate": "สง 2272", "conf": 59},
        {"plate": "สง 2277", "conf": 78},
        {"plate": "สง 2277", "conf": 82},
    ]

    consensus = lpr_consensus(observations)

    assert consensus == {
        "plate": "สง 2277",
        "key": "สง2277",
        "confidence": 80,
        "support": 2,
        "sample_count": 3,
    }
    assert lpr_review_status(observations) == "verified"


def test_lpr_review_waits_for_consensus_and_marks_split_votes_uncertain():
    assert lpr_review_status([{"plate": "สง 2272", "conf": 59}]) == "pending"
    assert lpr_review_status([{"plate": "สง 2277", "conf": 91}]) == "verified"

    split_observations = [
        {"plate": "สง 2272", "conf": 59},
        {"plate": "สง 2277", "conf": 78},
        {"plate": "สง 2271", "conf": 66},
        {"plate": "สง 2272", "conf": 62},
        {"plate": "สง 2277", "conf": 75},
    ]
    assert lpr_review_status(split_observations) == "uncertain"


def test_lpr_partial_plate_is_never_marked_verified():
    # Digits-only partial reads like "5762" must never be verified, even at 99% confidence
    partial_obs = [{"plate": "5762", "conf": 99}]
    assert lpr_review_status(partial_obs) == "pending"

    # When attempts are exhausted (max_samples reached), status should be partial, not verified or uncertain
    max_partial_obs = [{"plate": "5762", "conf": 95}] * 5
    assert lpr_review_status(max_partial_obs) == "partial"


def test_lpr_consensus_prefers_complete_plate_over_partial_on_tie():
    # If one frame only caught digits "5762" and another caught complete "1กฒ 5762"
    observations = [
        {"plate": "5762", "conf": 90},
        {"plate": "1กฒ 5762", "conf": 85},
    ]
    consensus = lpr_consensus(observations)
    assert consensus["plate"] == "1กฒ 5762"



def test_thai_ocr_digits_are_normalized_to_arabic_digits():
    assert normalize_thai_plate_digits("กง 7๖0๖") == "กง 7606"
    assert normalize_thai_plate_digits("๑๒๓๔๕๖๗๘๙๐") == "1234567890"


def test_lpr_quality_prefers_sharp_large_non_clipped_vehicle_crop():
    sharp = np.zeros((120, 200, 3), dtype=np.uint8)
    sharp[:, ::4] = 255
    blurred = np.full((120, 200, 3), 127, dtype=np.uint8)

    sharp_score = lpr_frame_quality_score(sharp)
    blurred_score = lpr_frame_quality_score(blurred)
    clipped_score = lpr_frame_quality_score(sharp, touches_frame_edge=True)

    assert sharp_score > blurred_score
    assert clipped_score < sharp_score


def test_lpr_upscale_factor_is_dynamic_and_bounded():
    assert lpr_upscale_factor(300, 200) == 2.0
    assert lpr_upscale_factor(120, 80) == 3.0
    assert lpr_upscale_factor(60, 40) == 4.0
    assert lpr_upscale_factor(2000, 800) == 0.75


def test_prepare_lpr_images_keeps_more_than_lower_half():
    vehicle = np.zeros((100, 160, 3), dtype=np.uint8)
    prepared = prepare_lpr_ocr_images(vehicle)

    assert prepared["roi_y"] == 22
    assert prepared["roi"].shape == (78, 160, 3)
    assert len(prepared["ocr_images"]) == 3
    assert all(image.shape[0] > prepared["roi"].shape[0] for image in prepared["ocr_images"])

    # Motorcycle keeps full vertical height or bottom 90%
    mc = np.zeros((100, 80, 3), dtype=np.uint8)
    prepared_mc = prepare_lpr_ocr_images(mc, is_motorcycle=True)
    assert prepared_mc["roi_y"] == 0


def test_expanded_plate_bbox_preserves_upper_row_and_border():
    # OCR found only a number row around x=60..140, y=70..90 on a 2x image.
    points = [[60, 70], [140, 70], [140, 90], [60, 90]]
    x1, y1, x2, y2 = expanded_plate_bbox(points, scale=2.0, image_width=120, image_height=70)

    # Source text box would be x=30..70, y=35..45. Crop must include generous
    # context above it for Thai letters/province plus the physical plate edge.
    assert x1 <= 18
    assert x2 >= 82
    assert y1 <= 23
    assert y2 >= 49
    assert (x2 - x1) >= 64
    assert (y2 - y1) >= 28


def test_expanded_plate_bbox_centres_the_text_and_keeps_province_space():
    # A wide OCR result usually represents only the large registration row.
    points = [[100, 80], [300, 80], [300, 120], [100, 120]]
    x1, y1, x2, y2 = expanded_plate_bbox(
        points,
        scale=2.0,
        image_width=220,
        image_height=120,
    )

    text_x1, text_y1, text_x2, text_y2 = 50, 40, 150, 60
    left_context = text_x1 - x1
    right_context = x2 - text_x2
    top_context = text_y1 - y1
    bottom_context = y2 - text_y2

    assert abs(left_context - right_context) <= 1
    assert bottom_context >= top_context - 1
    assert top_context >= (text_y2 - text_y1)
    assert bottom_context >= (text_y2 - text_y1)
    assert (x2 - x1) / (y2 - y1) <= 2.5


def test_expanded_plate_bbox_shifts_full_window_inside_image_edges():
    points = [[340, 100], [400, 100], [400, 130], [340, 130]]
    x1, y1, x2, y2 = expanded_plate_bbox(
        points,
        scale=2.0,
        image_width=200,
        image_height=100,
    )

    assert x2 == 200
    assert (x2 - x1) >= 64
    assert 0 <= y1 < y2 <= 100


def test_expanded_plate_bbox_clamps_at_image_edges():
    points = [[0, 0], [30, 0], [30, 20], [0, 20]]
    box = expanded_plate_bbox(points, scale=1.0, image_width=50, image_height=30)
    assert box[0] == 0
    assert box[1] == 0
    assert 0 < box[2] <= 50
    assert 0 < box[3] <= 30


def test_commercial_truck_plates_are_recognized_as_complete():
    assert plate_read_kind("80-4058") == "complete"
    assert plate_read_kind("70-1234") == "complete"
    assert plate_read_kind("804058") == "complete"
    assert plate_read_kind("80-12") == "complete"
    assert plate_read_kind("5762") == "partial"
    assert plate_read_kind("123") == "partial"
    assert plate_read_kind("กข 1234") == "complete"
    assert plate_read_kind("1กข 1234") == "complete"
