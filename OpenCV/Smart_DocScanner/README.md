# Smart Document Scanner

A computer vision project built with Python and OpenCV that automatically detects a document in an image, corrects its perspective, and generates a clean scanned version.

## Features

- Document boundary detection
- Canny edge detection
- Contour detection
- Four-corner document detection
- Perspective transformation
- Grayscale conversion
- Adaptive thresholding
- Automatic scanned document generation

## Tech Stack

- Python
- OpenCV
- NumPy

## How It Works

```text
Input Image
    ↓
Grayscale
    ↓
Gaussian Blur
    ↓
Canny Edge Detection
    ↓
Contour Detection
    ↓
Four-Corner Detection
    ↓
Perspective Transformation
    ↓
Adaptive Thresholding
    ↓
Scanned Document

Project Structure
Smart_DocScanner/
├── scanner.py
├── input/
│   └── document.jpg
├── output/
│   └── scanned_document.jpg
├── requirements.txt
└── README.md

Installation

Install the required dependencies:

pip install -r requirements.txt
Run
python scanner.py

The final scanned document is saved as:

output/scanned_document.jpg
