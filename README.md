# Image Filter Studio

A Python image processing application with a Streamlit interface.

The project allows you to load an image, apply several visual filters in real time, compare the result with the original, and then export the modified image.

## Features

- Grayscale conversion
- Edge detection
- Gaussian blur
- Sepia effect
- Vintage effect
- Brightness adjustment
- Contrast adjustment
- Before/After preview
- PNG export

## Technical Stack

- Python
- OpenCV
- NumPy
- Pillow
- Streamlit

## Project Structure

```text
image-filter-app/
├── app.py
├── filters.py
├── requirements.txt
├── README.md
└── tests/
└── test_filters.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Launch the application

```bash
streamlit run app.py

```

Then open the URL displayed by Streamlit in your browser.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Use Cases

- Quick image editing
- Generation of simple visual effects
- Mini computer vision project
- OpenCV demonstration with a lightweight web interface

## Possible Improvements

- Adding more filters
- Change history
- Batch processing
- Saving in multiple formats
