# Sunny Portrait Retouch

A Python/OpenCV portrait grading script tuned for the original `IMG_6224(2).jpeg` image.

It combines a sunny, pale-blue sky treatment with warm color grading; mild natural skin smoothing, spot retouching, and subtle jawline lightening; pinkish-red lips, rosy cheeks, creamy reflector highlights, subtle bouquet saturation, added local detail, and a soft glow.

**The output keeps the same width and height in pixels as the input.** JPEG compression may change the underlying pixel values and metadata. This is a local image processing workflow, not a Photoshop/Lightroom preset or a generative model.

## Installation

You need Python 3.10+.

```bash
python -m pip install -r requirements.txt
```

## Usage

```bash
python portrait_grade.py "IMG_6224(2).jpeg" -o retouched.jpg
```

To adjust the overall strength:

```bash
python portrait_grade.py "IMG_6224(2).jpeg" -o retouched.jpg --strength 0.8
```

The `--strength` setting supports a value between 0.0 and 1.8 (higher numbers make the changes more noticeable).

## Important notes

- The face, lips, sky, and bouquet masks use normalized positions calibrated to the *specific portrait* named above. They are not automatically detected for other photographs. Adjust the masks in `portrait_grade.py` to use another composition.
- The algorithm is an approximation of the retouching directions, not a lossless recreation of an earlier edited image.
- The sky gradient cannot recover detail that was irreversibly clipped in the source photograph.
- Keep private photos out of the Git repository. The `.gitignore` excludes common photo files by default.

## Run tests

```bash
python -m unittest discover -s tests -v
```
