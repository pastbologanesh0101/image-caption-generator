"""Unit tests for caption.py.

All test fixtures are synthetic images generated on the fly via
generate_test_images.py (solid colors, gradients, shapes, checkerboard) --
no real photos are used.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import caption
import generate_test_images as gti


class CaptionGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp(prefix="caption_test_images_")
        cls.paths = {
            os.path.splitext(name)[0]: path
            for name, path in zip(
                gti.DEMO_IMAGES.keys(), gti.generate_all(cls.tmp_dir)
            )
        }

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    # -- brightness -----------------------------------------------------

    def test_brightness_classifies_dark_image_as_dark(self):
        img = caption.load_image(self.paths["dark"])
        result = caption.analyze_brightness(img)
        self.assertEqual(result["label"], "dark")
        self.assertLess(result["mean_brightness"], 85)

    def test_brightness_classifies_bright_image_as_bright(self):
        img = caption.load_image(self.paths["bright"])
        result = caption.analyze_brightness(img)
        self.assertEqual(result["label"], "bright")
        self.assertGreater(result["mean_brightness"], 170)

    # -- dominant color ---------------------------------------------------

    def test_dominant_color_detection_on_solid_red_image(self):
        img = caption.load_image(self.paths["solid_red"])
        result = caption.analyze_dominant_colors(img, k=3)
        r, g, b = result["dominant_colors_rgb"][0]
        # Solid red-ish image (220, 40, 40) in RGB; allow for JPEG/PNG +
        # kmeans float tolerance.
        self.assertGreater(r, g + 50)
        self.assertGreater(r, b + 50)

    # -- orientation / aspect ratio ---------------------------------------

    def test_orientation_classification_landscape_portrait_square(self):
        landscape = caption.analyze_orientation(caption.load_image(self.paths["landscape"]))
        portrait = caption.analyze_orientation(caption.load_image(self.paths["portrait"]))
        square = caption.analyze_orientation(caption.load_image(self.paths["square"]))

        self.assertEqual(landscape["label"], "landscape")
        self.assertEqual(portrait["label"], "portrait")
        self.assertEqual(square["label"], "square")

    # -- face detection -----------------------------------------------------

    def test_face_detection_returns_zero_on_image_with_no_faces(self):
        img = caption.load_image(self.paths["shapes"])
        result = caption.detect_faces(img)
        self.assertEqual(result["count"], 0)
        self.assertEqual(result["positions"], [])

    # -- edge density ---------------------------------------------------

    def test_edge_density_distinguishes_blank_from_checkerboard(self):
        blank_img = caption.load_image(self.paths["blank"])
        checker_img = caption.load_image(self.paths["checkerboard"])

        blank_result = caption.analyze_edge_density(blank_img)
        checker_result = caption.analyze_edge_density(checker_img)

        self.assertEqual(blank_result["label"], "simple")
        self.assertIn(checker_result["label"], ("busy", "moderately detailed"))
        self.assertGreater(
            checker_result["edge_pixel_ratio"], blank_result["edge_pixel_ratio"]
        )

    # -- caption generation -----------------------------------------------

    def test_caption_generation_produces_nonempty_sentence(self):
        img = caption.load_image(self.paths["gradient"])
        features = caption.analyze_image(img)
        result = caption.generate_caption(features)

        self.assertIsInstance(result, str)
        self.assertGreater(len(result.strip()), 0)
        self.assertTrue(result.endswith("."))
        # Should mention the detected orientation label somewhere.
        self.assertIn(features.orientation["label"], result)

    def test_caption_mentions_face_count_phrase(self):
        img = caption.load_image(self.paths["shapes"])
        features = caption.analyze_image(img)
        result = caption.generate_caption(features)
        self.assertIn("no faces detected", result)

    # -- error handling -----------------------------------------------------

    def test_missing_image_file_raises_clear_error(self):
        missing_path = os.path.join(self.tmp_dir, "does_not_exist.jpg")
        with self.assertRaises(FileNotFoundError) as ctx:
            caption.load_image(missing_path)
        self.assertIn("not found", str(ctx.exception).lower())

    def test_corrupt_image_file_raises_clear_error(self):
        corrupt_path = os.path.join(self.tmp_dir, "corrupt.jpg")
        with open(corrupt_path, "wb") as f:
            f.write(b"this is not a valid image file, just plain text bytes")

        with self.assertRaises(ValueError) as ctx:
            caption.load_image(corrupt_path)
        self.assertIn("could not decode", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
