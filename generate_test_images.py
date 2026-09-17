#!/usr/bin/env python3
"""
Generate small synthetic test/demo images using PIL only.

These images are entirely procedurally generated (solid colors, simple
shapes, gradients) -- no real photos are downloaded or scraped. They are
used both for the demo output shown in README.md and as fixtures for the
unit tests in tests/test_caption.py.

Usage:
    python generate_test_images.py [output_dir]
"""

from __future__ import annotations

import os
import sys

from PIL import Image, ImageDraw


def make_solid_color_image(color: tuple[int, int, int], size=(120, 120)) -> Image.Image:
    return Image.new("RGB", size, color)


def make_dark_image(size=(120, 120)) -> Image.Image:
    return Image.new("RGB", size, (10, 10, 12))


def make_bright_image(size=(120, 120)) -> Image.Image:
    return Image.new("RGB", size, (245, 245, 240))


def make_gradient_image(size=(160, 120)) -> Image.Image:
    img = Image.new("RGB", size)
    draw = ImageDraw.Draw(img)
    width, height = size
    for x in range(width):
        t = x / max(1, width - 1)
        r = int(255 * t)
        g = int(80 + 40 * t)
        b = int(255 * (1 - t))
        draw.line([(x, 0), (x, height)], fill=(r, g, b))
    return img


def make_landscape_image(size=(200, 100)) -> Image.Image:
    return Image.new("RGB", size, (180, 120, 60))


def make_portrait_image(size=(100, 200)) -> Image.Image:
    return Image.new("RGB", size, (60, 90, 180))


def make_square_image(size=(150, 150)) -> Image.Image:
    return Image.new("RGB", size, (100, 180, 100))


def make_checkerboard_image(size=(120, 120), square=10) -> Image.Image:
    """A busy/high-contrast pattern used to test edge-density detection."""
    img = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    width, height = size
    for y in range(0, height, square):
        for x in range(0, width, square):
            if ((x // square) + (y // square)) % 2 == 0:
                draw.rectangle([x, y, x + square, y + square], fill=(0, 0, 0))
    return img


def make_blank_image(size=(120, 120)) -> Image.Image:
    """A flat, featureless image -- the opposite of the checkerboard,
    used to test that edge density is low."""
    return Image.new("RGB", size, (128, 128, 128))


def make_shapes_image(size=(160, 160)) -> Image.Image:
    img = Image.new("RGB", size, (240, 240, 240))
    draw = ImageDraw.Draw(img)
    draw.ellipse([20, 20, 80, 80], fill=(220, 50, 50))
    draw.rectangle([90, 90, 150, 150], fill=(50, 50, 220))
    return img


DEMO_IMAGES = {
    "solid_red.png": lambda: make_solid_color_image((220, 40, 40)),
    "solid_blue.png": lambda: make_solid_color_image((40, 60, 220)),
    "dark.png": make_dark_image,
    "bright.png": make_bright_image,
    "gradient.png": make_gradient_image,
    "landscape.png": make_landscape_image,
    "portrait.png": make_portrait_image,
    "square.png": make_square_image,
    "checkerboard.png": make_checkerboard_image,
    "blank.png": make_blank_image,
    "shapes.png": make_shapes_image,
}


def generate_all(output_dir: str) -> list[str]:
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for name, factory in DEMO_IMAGES.items():
        img = factory()
        path = os.path.join(output_dir, name)
        img.save(path)
        paths.append(path)
    return paths


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "demo_images"
    created = generate_all(out_dir)
    for p in created:
        print(f"Created {p}")
