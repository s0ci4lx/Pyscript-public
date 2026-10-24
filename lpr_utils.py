"""Small, testable helpers for license-plate OCR and evidence crops."""

from __future__ import annotations

import re

import cv2
import numpy as np


THAI_TO_ARABIC_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")


def normalize_thai_plate_digits(text) -> str:
    """Use Arabic digits consistently for Thai registration plates."""
    if text is None:
        return ""
    return str(text).translate(THAI_TO_ARABIC_DIGITS)


def lpr_plate_key(text) -> str:
    """Return a stable comparison key without changing the displayed plate."""
    normalized = normalize_thai_plate_digits(text)
    return re.sub(r"[^ก-ฮ0-9a-zA-Z]", "", normalized).lower()


def plate_read_kind(text):
    """Conservative format gate, not proof that the text came from a plate."""
    text = normalize_thai_plate_digits(text).strip()
    key = re.sub(r"\s+", "", text)
    if re.fullmatch(r"[0-9]?[ก-ฮ]{1,3}[0-9]{1,4}", key):
        return "complete"
    # Thai commercial vehicles (trucks, trailers, buses): 2-digit class (10-99) + 1-4 digits
    if re.fullmatch(r"[1-9][0-9]-[0-9]{1,4}", key):
        return "complete"
    if re.fullmatch(r"[1-9][0-9][0-9]{3,4}", key):
        return "complete"
    if re.fullmatch(r"[0-9]{1,4}", key):
        return "partial"
    return "invalid"


def lpr_consensus(observations):
    """Select a plate from independent frame observations.

    Votes decide first, while mean OCR confidence breaks ties.  Keeping whole
    plate strings together avoids manufacturing a plate that no frame saw.
    """
    groups = {}
    for observation in observations or ():
        plate = str(observation.get("plate", "")).strip()
        key = lpr_plate_key(plate)
        if not key:
            continue
        confidence = max(0, min(100, int(round(observation.get("conf", 0) or 0))))
        group = groups.setdefault(key, {"key": key, "count": 0, "total_conf": 0, "best": None})
        group["count"] += 1
        group["total_conf"] += confidence
        if group["best"] is None or confidence > group["best"]["conf"]:
            group["best"] = {"plate": plate, "conf": confidence}

    if not groups:
        return None

    winner = max(
        groups.values(),
        key=lambda group: (
            group["count"],
            1 if plate_read_kind(group["best"]["plate"]) == "complete" else 0,
            group["total_conf"] / group["count"],
            group["best"]["conf"],
        ),
    )
    return {
        "plate": winner["best"]["plate"],
        "key": winner["key"],
        "confidence": int(round(winner["total_conf"] / winner["count"])),
        "support": winner["count"],
        "sample_count": sum(group["count"] for group in groups.values()),
    }


def lpr_review_status(
    observations,
    *,
    high_confidence: int = 70,
    min_consensus_samples: int = 3,
    max_samples: int = 5,
) -> str:
    """Classify a multi-frame OCR result as pending, verified, or uncertain."""
    consensus = lpr_consensus(observations)
    if consensus is None:
        return "pending"
    if plate_read_kind(consensus["plate"]) == "complete":
        if (
            consensus["sample_count"] >= min_consensus_samples
            and consensus["support"] > consensus["sample_count"] / 2
            and consensus["confidence"] >= 70
        ):
            return "verified"
        if (
            consensus["sample_count"] == 2
            and consensus["support"] == 2
            and consensus["confidence"] >= 65
        ):
            return "verified"
        if (
            consensus["sample_count"] >= 1
            and consensus["confidence"] >= high_confidence
            and consensus["support"] == consensus["sample_count"]
        ):
            return "verified"
    if consensus["sample_count"] >= max_samples:
        return "partial" if plate_read_kind(consensus["plate"]) == "partial" else "uncertain"
    return "pending"


def lpr_frame_quality_score(vehicle_crop: np.ndarray, touches_frame_edge: bool = False) -> float:
    """Rank vehicle crops for OCR using usable size, sharpness and edge clipping."""
    if vehicle_crop is None or vehicle_crop.size == 0:
        return 0.0

    height, width = vehicle_crop.shape[:2]
    if height <= 0 or width <= 0:
        return 0.0

    sample = vehicle_crop
    if width > 320:
        scale = 320.0 / width
        sample = cv2.resize(
            vehicle_crop,
            (320, max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness_factor = min(2.0, max(0.25, sharpness / 80.0))
    edge_factor = 0.55 if touches_frame_edge else 1.0
    return float(width * height) * sharpness_factor * edge_factor


def lpr_upscale_factor(
    width: int,
    height: int,
    target_short_side: int = 240,
    max_long_side: int = 960,
) -> float:
    """Return an OCR scale that enlarges distant crops without exploding large ones."""
    short_side = max(1, min(int(width), int(height)))
    long_side = max(1, max(int(width), int(height)))
    desired = max(2.0, min(4.0, target_short_side / short_side))
    long_side_limit = max(0.75, float(max_long_side) / long_side)
    return float(max(0.75, min(desired, long_side_limit)))


def prepare_lpr_ocr_images(vehicle_crop: np.ndarray, is_motorcycle: bool = False):
    """Build a generous vehicle ROI and complementary OCR representations.

    Thai plates on motorcycles, pickups and steep CCTV views can sit higher,
    so motorcycles use the full vertical crop (or bottom 90%), and cars retain
    the bottom 78% (roi_y = 0.22) rather than cutting off at 38%.
    """
    if vehicle_crop is None or vehicle_crop.size == 0:
        return None

    vehicle_h, vehicle_w = vehicle_crop.shape[:2]
    if is_motorcycle or (vehicle_w / max(1, vehicle_h) < 1.25):
        roi_y = int(vehicle_h * 0.10) if vehicle_h > 150 else 0
    else:
        roi_y = int(vehicle_h * 0.22)

    roi = vehicle_crop[roi_y:vehicle_h, 0:vehicle_w]
    if roi.size == 0:
        return None

    roi_h, roi_w = roi.shape[:2]
    scale = lpr_upscale_factor(roi_w, roi_h)
    interpolation = cv2.INTER_CUBIC if scale >= 1.0 else cv2.INTER_AREA
    upscaled = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=interpolation)

    ycrcb = cv2.cvtColor(upscaled, cv2.COLOR_BGR2YCrCb)
    channels = list(cv2.split(ycrcb))
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    channels[0] = clahe.apply(channels[0])
    enhanced = cv2.cvtColor(cv2.merge(channels), cv2.COLOR_YCrCb2BGR)
    enhanced = cv2.bilateralFilter(enhanced, 5, 40, 40)

    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    return {
        "roi": roi,
        "roi_y": roi_y,
        "scale": scale,
        "ocr_images": (enhanced, upscaled, gray_bgr),
    }


def expanded_plate_bbox(
    points,
    scale: float,
    image_width: int,
    image_height: int,
):
    """Map OCR points to a centred crop that includes the physical plate.

    EasyOCR often returns only the large registration row, not the smaller
    province row or the plate border.  Build a plate-shaped window around the
    text instead of independently padding each edge.  This keeps the detected
    text near the centre and, importantly for Thai plates, reserves slightly
    more room below the main row for the province text.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if pts.size == 0:
        return 0, 0, int(image_width), int(image_height)

    safe_scale = max(float(scale), 1e-6)
    min_x, min_y = np.min(pts, axis=0) / safe_scale
    max_x, max_y = np.max(pts, axis=0) / safe_scale
    text_w = max(1.0, float(max_x - min_x))
    text_h = max(1.0, float(max_y - min_y))

    # A single OCR row needs substantially more vertical context than a group
    # which already spans two rows.  In both cases, keep a useful minimum size
    # so a short/partial OCR result does not create a tiny corner crop.
    text_aspect = text_w / text_h
    if text_aspect >= 2.0:
        # Car plates (wide aspect): provide generous padding on both sides and below for province
        crop_w = max(72.0, text_w * 1.65, text_h * 4.8)
        crop_h = max(42.0, text_h * 3.50, crop_w / 2.15)
        centre_y = ((min_y + max_y) / 2.0) + (text_h * 0.08)
    else:
        # Motorcycle square plate or multi-row detection
        crop_w = max(68.0, text_w * 1.50, text_h * 2.0)
        crop_h = max(40.0, text_h * 1.60, crop_w / 1.75)
        centre_y = ((min_y + max_y) / 2.0) + (text_h * 0.05)

    centre_x = (min_x + max_x) / 2.0
    x1, x2 = _centered_interval(centre_x, crop_w, int(image_width))
    y1, y2 = _centered_interval(centre_y, crop_h, int(image_height))
    return x1, y1, x2, y2


def _centered_interval(center: float, requested_size: float, limit: int):
    """Return a fixed-size interval, shifting it intact when it meets an edge."""
    limit = max(0, int(limit))
    if limit == 0:
        return 0, 0

    size = min(limit, max(1, int(np.ceil(requested_size))))
    start = int(np.floor(float(center) - (size / 2.0)))
    end = start + size
    if start < 0:
        end -= start
        start = 0
    if end > limit:
        start -= end - limit
        end = limit
    start = max(0, start)
    return start, end


def plate_matches_target(observed, target) -> bool:
    """Match a configured full or partial plate; empty targets never match.

    Digit-only targets match within a contiguous number group, never across
    Thai registration letters (e.g. 1กข234 must not match target 1234).
    """
    observed_key = lpr_plate_key(observed)
    target_key = lpr_plate_key(target)
    if not observed_key or not target_key:
        return False
    if target_key.isdigit():
        return any(target_key in group for group in re.findall(r"[0-9]+", observed_key))
    return target_key in observed_key
