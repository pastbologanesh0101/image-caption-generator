#!/usr/bin/env python3
"""
AI Image Caption Generator (Classical Computer Vision Edition)
================================================================

Generates a natural-language caption for an image using *classical*
(non-deep-learning) computer vision heuristics: color histograms /
k-means for dominant colors, pixel-intensity statistics for
brightness, aspect ratio for orientation, OpenCV's bundled Haar
cascade classifier for face detection, and Canny edge counts for
scene complexity. See README.md for why this project deliberately
avoids neural image-captioning models.

Usage:
    python caption.py path/to/image.jpg
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image(path: str) -> np.ndarray:
    """Load an image from disk as a BGR numpy array.

    Raises FileNotFoundError if the path does not exist, and ValueError
    if the file exists but cannot be decoded as an image (e.g. corrupt
    data or an unsupported/non-image file).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")

    if not os.path.isfile(path):
        raise ValueError(f"Path is not a file: {path}")

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(
            f"Could not decode image (file may be corrupt or not a "
            f"supported image format): {path}"
        )
    return img


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def analyze_brightness(img: np.ndarray) -> dict[str, Any]:
    """Classify overall exposure as dark / well-lit / bright."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))

    if mean_brightness < 85:
        label = "dark"
    elif mean_brightness > 170:
        label = "bright"
    else:
        label = "well-lit"

    return {"mean_brightness": mean_brightness, "label": label}


def analyze_orientation(img: np.ndarray) -> dict[str, Any]:
    """Classify the image as portrait / landscape / square."""
    height, width = img.shape[:2]
    aspect_ratio = width / height

    if abs(aspect_ratio - 1.0) < 0.05:
        label = "square"
    elif aspect_ratio > 1.0:
        label = "landscape"
    else:
        label = "portrait"

    return {"width": width, "height": height, "aspect_ratio": aspect_ratio, "label": label}


def analyze_dominant_colors(img: np.ndarray, k: int = 3) -> dict[str, Any]:
    """Find dominant colors via k-means clustering on pixel colors.

    Returns colors as RGB tuples (converted from OpenCV's native BGR)
    ordered from most to least dominant by cluster size.
    """
    # Downsample for speed on large images.
    small = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)
    pixels = small.reshape(-1, 3).astype(np.float32)

    k = max(1, min(k, len(np.unique(pixels.view(np.dtype((np.void, pixels.dtype.itemsize * 3))), axis=0))))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _compactness, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS
    )

    labels = labels.flatten()
    counts = np.bincount(labels, minlength=k)
    order = np.argsort(-counts)

    dominant_colors_rgb = []
    for idx in order:
        b, g, r = centers[idx]
        dominant_colors_rgb.append((int(r), int(g), int(b)))

    return {"dominant_colors_rgb": dominant_colors_rgb, "counts": counts[order].tolist()}


def analyze_color_temperature(img: np.ndarray) -> dict[str, Any]:
    """Classify the overall color temperature as warm / cool / neutral
    by comparing mean red/orange-ish channel weight vs blue channel weight.
    """
    b, g, r = cv2.split(img.astype(np.float32))
    mean_r = float(np.mean(r))
    mean_g = float(np.mean(g))
    mean_b = float(np.mean(b))

    warm_score = mean_r - mean_b
    if warm_score > 15:
        label = "warm"
    elif warm_score < -15:
        label = "cool"
    else:
        label = "neutral"

    return {"mean_r": mean_r, "mean_g": mean_g, "mean_b": mean_b, "warm_score": warm_score, "label": label}


def _cascade_path() -> str:
    return os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")


def detect_faces(img: np.ndarray) -> dict[str, Any]:
    """Detect faces using OpenCV's bundled Haar cascade classifier.

    Returns the count of detected faces and their bounding boxes, plus
    a rough description of where the faces sit in the frame (this uses
    zero external downloads -- the cascade XML ships inside opencv-python).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(_cascade_path())

    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    height, width = gray.shape[:2]
    positions = []
    for (x, y, w, h) in faces:
        cx, cy = x + w / 2, y + h / 2
        horiz = "left" if cx < width / 3 else "right" if cx > 2 * width / 3 else "center"
        vert = "top" if cy < height / 3 else "bottom" if cy > 2 * height / 3 else "middle"
        positions.append(f"{vert}-{horiz}")

    return {"count": int(len(faces)), "positions": positions}


def analyze_edge_density(img: np.ndarray) -> dict[str, Any]:
    """Classify scene complexity (simple vs busy) using Canny edge density."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_pixel_ratio = float(np.count_nonzero(edges)) / edges.size

    if edge_pixel_ratio < 0.02:
        label = "simple"
    elif edge_pixel_ratio > 0.12:
        label = "busy"
    else:
        label = "moderately detailed"

    return {"edge_pixel_ratio": edge_pixel_ratio, "label": label}


_COLOR_NAMES = [
    ("red", (200, 60, 60)),
    ("orange", (220, 130, 60)),
    ("yellow", (210, 200, 70)),
    ("green", (70, 160, 80)),
    ("cyan", (70, 170, 180)),
    ("blue", (60, 90, 200)),
    ("purple", (140, 80, 180)),
    ("pink", (220, 130, 170)),
    ("white", (235, 235, 235)),
    ("gray", (128, 128, 128)),
    ("black", (25, 25, 25)),
]


def _nearest_color_name(rgb: tuple[int, int, int]) -> str:
    r, g, b = rgb
    best_name, best_dist = "gray", float("inf")
    for name, (cr, cg, cb) in _COLOR_NAMES:
        dist = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name


@dataclass
class ImageFeatures:
    brightness: dict = field(default_factory=dict)
    orientation: dict = field(default_factory=dict)
    colors: dict = field(default_factory=dict)
    temperature: dict = field(default_factory=dict)
    faces: dict = field(default_factory=dict)
    edges: dict = field(default_factory=dict)


def analyze_image(img: np.ndarray) -> ImageFeatures:
    """Run the full classical-CV feature-extraction pipeline on an image."""
    return ImageFeatures(
        brightness=analyze_brightness(img),
        orientation=analyze_orientation(img),
        colors=analyze_dominant_colors(img),
        temperature=analyze_color_temperature(img),
        faces=detect_faces(img),
        edges=analyze_edge_density(img),
    )


# ---------------------------------------------------------------------------
# Caption synthesis
# ---------------------------------------------------------------------------

def generate_caption(features: ImageFeatures) -> str:
    """Combine extracted features into a readable natural-language caption
    using a template-based synthesis approach (no language model involved).
    """
    brightness_label = features.brightness.get("label", "well-lit")
    orientation_label = features.orientation.get("label", "landscape")
    temperature_label = features.temperature.get("label", "neutral")
    edges_label = features.edges.get("label", "moderately detailed")
    face_count = features.faces.get("count", 0)

    dominant_rgb = features.colors.get("dominant_colors_rgb", [])
    dominant_name = _nearest_color_name(dominant_rgb[0]) if dominant_rgb else "neutral-toned"

    if face_count == 0:
        face_phrase = "no faces detected"
    elif face_count == 1:
        face_phrase = "featuring 1 face in the frame"
    else:
        face_phrase = f"featuring {face_count} faces in the frame"

    temperature_phrase = f"{temperature_label} tones" if temperature_label != "neutral" else "neutral tones"

    sentence = (
        f"A {brightness_label}, {edges_label} {orientation_label}-oriented image "
        f"with predominantly {dominant_name} and {temperature_phrase}, {face_phrase}."
    )

    return sentence


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def format_report(path: str, features: ImageFeatures, caption: str) -> str:
    lines = [
        f"Image: {path}",
        "",
        f"Caption: {caption}",
        "",
        "Detected features:",
        f"  Brightness:      {features.brightness}",
        f"  Orientation:     {features.orientation}",
        f"  Dominant colors: {features.colors.get('dominant_colors_rgb')}",
        f"  Color temp:      {features.temperature}",
        f"  Faces:           {features.faces}",
        f"  Edge density:    {features.edges}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a classical-CV image caption (no deep learning)."
    )
    parser.add_argument("image", help="Path to the input image file")
    args = parser.parse_args(argv)

    try:
        img = load_image(args.image)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    features = analyze_image(img)
    caption = generate_caption(features)
    print(format_report(args.image, features, caption))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
