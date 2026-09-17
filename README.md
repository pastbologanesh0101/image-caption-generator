# AI Image Caption Generator (Classical Computer Vision)

Generates a natural-language caption for an image using **classical, heuristic
computer vision** — not a deep learning / neural image-captioning model.

## Important: scope and honesty about the approach

This project does **not** use a neural network like BLIP, Show-and-Tell,
CLIP, or any other trained image-captioning model. Real neural captioning
models require downloading large pretrained weight files (hundreds of MB to
several GB) and a deep learning framework (PyTorch/TensorFlow), which is
overkill for a small, fully offline, dependency-light portfolio project.

Instead, this generator extracts a handful of **classical image-processing
features** using OpenCV, Pillow, and NumPy, and stitches them into a sentence
with a template system. It is intentionally simple and explainable — every
word in the output caption is traceable to a specific measurable feature of
the image. Think of it as "structured image description via heuristics,"
not "AI-generated free-text captioning."

## Detected features

| Feature | Technique |
|---|---|
| Dominant colors | k-means clustering (`cv2.kmeans`) on downsampled pixel colors, mapped to the nearest named color |
| Brightness / exposure | Mean pixel intensity on the grayscale image, bucketed into dark / well-lit / bright |
| Orientation | Width/height aspect ratio, bucketed into portrait / landscape / square |
| Faces | `cv2.CascadeClassifier` with OpenCV's bundled `haarcascade_frontalface_default.xml` (ships inside `opencv-python`, no internet download needed) — returns a count and rough position (e.g. "top-left") for each detected face |
| Scene complexity | Canny edge detection (`cv2.Canny`); the fraction of edge pixels is bucketed into simple / moderately detailed / busy |
| Color temperature | Mean red channel minus mean blue channel, bucketed into warm / cool / neutral |

## How captions are built

`generate_caption()` in `caption.py` is a pure template function: it plugs
the labels above into a sentence like

```
A {brightness}, {complexity} {orientation}-oriented image with predominantly
{dominant color} and {temperature} tones, {face phrase}.
```

No language model is involved in caption synthesis — it's string
formatting driven by measured feature values.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
# Generate a few synthetic demo/test images (solid colors, shapes, gradients)
python generate_test_images.py demo_images

# Caption one of them
python caption.py demo_images/landscape.png
```

### Example output

```
$ python caption.py demo_images/landscape.png
Image: demo_images/landscape.png

Caption: A well-lit, simple landscape-oriented image with predominantly orange and warm tones, no faces detected.

Detected features:
  Brightness:      {'mean_brightness': 131.0, 'label': 'well-lit'}
  Orientation:     {'width': 200, 'height': 100, 'aspect_ratio': 2.0, 'label': 'landscape'}
  Dominant colors: [(180, 120, 60)]
  Color temp:      {'mean_r': 180.0, 'mean_g': 120.0, 'mean_b': 60.0, 'warm_score': 120.0, 'label': 'warm'}
  Faces:           {'count': 0, 'positions': []}
  Edge density:    {'edge_pixel_ratio': 0.0, 'label': 'simple'}
```

```
$ python caption.py demo_images/checkerboard.png
Image: demo_images/checkerboard.png

Caption: A well-lit, busy square-oriented image with predominantly black and neutral tones, no faces detected.
```

## Test images

`generate_test_images.py` procedurally creates small PNGs with Pillow only
(solid colors, gradients, a checkerboard, simple shapes) — no real photos
are downloaded or scraped, so the repo has no external image dependencies
and the test suite runs fully offline.

## Running tests

```bash
pip install pytest
pytest -v tests/
```

The suite (`tests/test_caption.py`) covers:

- Brightness classification on synthetic all-dark and all-bright images
- Dominant color detection on a solid-color synthetic image
- Aspect-ratio/orientation classification (landscape / portrait / square)
- Face detection returning 0 on an image with no face-like pattern
- Edge-density classification distinguishing a blank image from a checkerboard
- Caption generation producing a non-empty, grammatical sentence that
  incorporates detected features
- Missing and corrupt image files raising clear, typed errors

CI (`.github/workflows/tests.yml`) runs the same suite on Python 3.11 and
3.12 on every push and pull request.

## Project structure

```
caption.py                 # feature extraction + caption synthesis + CLI
generate_test_images.py    # synthetic PIL-generated demo/test images
tests/test_caption.py      # unit tests
.github/workflows/tests.yml
requirements.txt
LICENSE
```

## Limitations

Because this is a heuristic, non-learned system:

- Dominant color naming is a nearest-neighbor match against a small fixed
  palette, not a perceptual color-naming model.
- Face detection is limited to what Haar cascades can find (frontal,
  reasonably well-lit faces); it will miss profiles, occluded faces, or
  faces in low contrast.
- Captions describe *visual properties*, not scene semantics — it will
  never say "a dog running on a beach," only things like "a bright,
  moderately detailed landscape-oriented image with warm tones."

## License

MIT — see [LICENSE](LICENSE).
