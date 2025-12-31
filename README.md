# Real-Time Sudoku Vision Solver

A computer vision–based prototype that detects a Sudoku grid from a live webcam feed and solves it using a classic backtracking algorithm.

This project focuses on **real-time grid detection, image preprocessing, and algorithmic problem solving**, with digit recognition planned as a future extension.

---

## Features

- Live webcam input using OpenCV
- Adaptive thresholding and contour detection
- Automatic Sudoku grid localization
- Grid segmentation into 9×9 cells
- Backtracking-based Sudoku solver
- Real-time solution overlay on video feed

---

## How It Works

1. Captures frames from the webcam
2. Applies grayscale conversion, blurring, and adaptive thresholding
3. Detects the largest contour as the Sudoku grid
4. Extracts and normalizes the grid region
5. Splits the grid into 81 cells
6. Solves the Sudoku using a recursive backtracking algorithm
7. Overlays the solution onto the live video feed

---

## Current Limitations

- Digit recognition is **not yet implemented**
- Perspective correction (homography) is not applied
- Works best with clearly visible, front-facing Sudoku grids

These limitations are intentional and documented as part of an iterative development approach.

---

## Planned Improvements

- CNN-based digit recognition
- Perspective transformation for skewed grids
- Grid line removal and cell normalization
- Model inference optimization for real-time performance

---

## Tech Stack

- Python
- OpenCV
- NumPy

---

## Usage

```bash
pip install opencv-python numpy
python main.py

