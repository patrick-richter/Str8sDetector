# Str8ts Detector

## Overview

The **Str8ts Detector** is a Python-based tool for detecting and processing Str8ts puzzle grids from images. It uses computer vision and OCR (Optical Character Recognition) techniques to:

1. Detect the corners of the Str8ts puzzle grid in the input image.
2. Apply a perspective transformation to generate a top-down view of the grid.
3. Extract individual cells from the transformed grid.
4. Analyze the color (black or white) of each cell.
5. Recognize numbers in the cells using OCR.
6. Reconstruct the full Str8ts grid with cell colors and numbers.

The tool is designed to be modular, efficient, and easily extendable for further processing or integration with other puzzle-solving algorithms.

## Key Features

- **Corner Detection**: Uses contour detection and polygon approximation to identify the four corners of the puzzle grid.
- **Perspective Transformation**: Ensures a normalized, top-down view of the grid for accurate cell extraction.
- **Cell Analysis**: Distinguishes between black and white cells based on pixel intensity.
- **OCR Integration**: Leverages the `RapidOCR` library for fast and reliable text recognition.
- **Output Visualization**: Saves intermediate results, including corner visualization, warped grid, and processed cells.

## Chosen OCR Library

The project uses **RapidOCR**, an efficient and lightweight OCR library based on ONNX Runtime. RapidOCR was selected for its:

- High accuracy in recognizing digits and text.
- Speed and low resource consumption.
- Ease of integration with Python-based workflows.

RapidOCR handles both the initial OCR pass and an additional inverted cell OCR pass to improve recognition in black cells.

## Grid Detection Algorithm

The grid detection algorithm consists of the following steps:

1. **Preprocessing**:

   - Converts the input image to grayscale.
   - Applies adaptive thresholding to emphasize grid lines.
   - Performs morphological operations to close gaps and remove noise.

2. **Contour Detection**:

   - Finds external contours in the preprocessed image.
   - Selects the largest contour, assumed to be the puzzle grid.

3. **Corner Approximation**:

   - Approximates the contour to a quadrilateral using the Douglas-Peucker algorithm.
   - Extracts the four corners of the grid.

4. **Perspective Transformation**:
   - Uses OpenCV's `getPerspectiveTransform` to warp the grid to a standardized size for easier cell extraction and analysis.

## Requirements

- **Python 3.11** is required for this project. You can check if Python 3.11 is installed by running:

  ```bash
  python3.11 --version
  ```

  If you don't have Python 3.11 installed, download it from the official Python website:
  https://www.python.org/downloads/

## Setting Up the Virtual Environment

1. **Create the virtual environment**:

   Once Python 3.11 is installed, create a virtual environment inside the `.venv` folder:

   ```bash
   python3.11 -m venv .venv
   ```

2. **Activate the virtual environment**:

   After creating the virtual environment, activate it:

   - For Linux/MacOS:

     ```bash
     source .venv/bin/activate
     ```

   - For Windows:

     ```bash
     .venv\Scripts\activate
     ```

3. **Install the project dependencies**:

   With the virtual environment activated, install the required packages using the `requirements.txt` file:

   ```bash
   pip install -r requirements.txt
   ```

4. **Deactivate the virtual environment**:

   Once you're done working in the virtual environment, you can deactivate it by running:

   ```bash
   deactivate
   ```

## How to Run the Project

1. **Activate the virtual environment**:

   Before running the script, ensure the virtual environment is activated:

   ```bash
   source .venv/bin/activate  # For Linux/MacOS
   .venv\Scripts\activate   # For Windows
   ```

2. **Run the detection script**:

   Use Python 3.11 to execute the script:

   ```bash
   python3.11 run_detection.py
   ```

3. **Change the input image**:

   You can specify a different input image by modifying the `image_path` variable at the bottom of the `run_detection.py` script:

   ```python
   if __name__ == "__main__":
       image_path = Path("path/to/your/image.jpeg")
       grid = process_image(image_path)
   ```

   Replace `"path/to/your/image.jpeg"` with the path to your desired image file.
