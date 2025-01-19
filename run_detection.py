"""
Str8ts Puzzle Solver

This module provides functionality to process Str8ts puzzle images, detect grid corners, 
perform perspective transformation, analyze cell colors, and extract puzzle data from OCR results.

Key Features:
- Detects grid corners in an image of a Str8ts puzzle.
- Applies perspective transformations for top-down grid views.
- Analyzes cell colors (black or white) and detects numbers using OCR.
- Extracts and processes individual cells for Str8ts grid reconstruction.

Dependencies:
- OpenCV (cv2)
- numpy
- rapidocr_onnxruntime

Author: Patrick Richter
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, TypedDict, NamedTuple
import time
import numpy as np
from numpy.typing import NDArray
from rapidocr_onnxruntime import RapidOCR

import cv2


class Corner(NamedTuple):
    x: int
    y: int


class Corners(TypedDict):
    top_left: Corner
    top_right: Corner
    bottom_left: Corner
    bottom_right: Corner


class CellColor(Enum):
    BLACK = "black"
    WHITE = "white"


@dataclass
class Cell:
    color: CellColor
    number: Optional[int]


@dataclass
class Str8tsGrid:
    cells: list[list[Cell]]  # 9x9 grid


@dataclass
class DetectionResult:
    corners: Corners
    visualization: NDArray[np.uint8]


def extract_grid_cells(warped: NDArray[np.uint8]) -> list[list[NDArray[np.uint8]]]:
    """
    Split the warped grid image into individual cells.

    Args:
        warped (NDArray[np.uint8]): Warped grid image (grayscale or color).

    Returns:
        list[list[NDArray[np.uint8]]]: 2D list of cell images.
    """
    height, width = warped.shape[:2]
    cell_height = height // 9
    cell_width = width // 9

    cells = []
    for i in range(9):
        row = []
        for j in range(9):
            # Calculate cell coordinates
            y1 = i * cell_height
            y2 = (i + 1) * cell_height
            x1 = j * cell_width
            x2 = (j + 1) * cell_width

            # Extract cell with some margin
            margin = 5
            cell = warped[
                max(y1 + margin, 0) : min(y2 - margin, height),
                max(x1 + margin, 0) : min(x2 - margin, width),
            ]
            row.append(cell)
        cells.append(row)

    return cells


def analyze_cell_color(cell: NDArray[np.uint8]) -> CellColor:
    """
    Determine if a cell is black or white based on pixel intensity.

    Args:
        cell (NDArray[np.uint8]): Cell image (grayscale or color).

    Returns:
        CellColor: BLACK or WHITE based on intensity threshold.
    """
    if len(cell.shape) == 3:
        cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

    # Calculate average intensity
    avg_intensity = np.mean(cell)
    threshold = 50  # May need tuning

    return CellColor.BLACK if avg_intensity < threshold else CellColor.WHITE


def detect_digit(
    cell_coords: Tuple[int, int, int, int], ocr_results: list
) -> Optional[int]:
    """
    Detect a single digit from a numpy array containing the image by checking OCR results
    for coordinates that fall within the cell boundaries.

    Args:
        cell_coords: (x1, y1, x2, y2) coordinates of the cell in the full grid
        ocr_results: List of OCR results with coordinates and detected text

    Returns:
        int or None: Detected digit (0-9) or None if no digit was detected
    """
    x1, y1, x2, y2 = cell_coords  # Unpack cell coordinates

    # Loop through OCR results and check if they intersect with the cell coordinates
    for result in ocr_results:
        coordinates, text, _ = result
        xmin, ymin = coordinates[0]
        xmax, ymax = coordinates[2]

        # Check if the detected OCR text falls within the cell's boundaries
        if x1 <= xmin <= xmax <= x2 and y1 <= ymin <= ymax <= y2:
            if text.isdigit():
                return int(text)  # Return the detected digit
            elif text.strip() in ["L", "T"]:
                return 1
    return None  # Return None if no digit was detected


def detect_grid_corners(image_path: Path) -> Optional[DetectionResult]:
    """
    Detect the corners of a Str8ts puzzle grid in an image.

    Args:
        image_path: Path to the input image file

    Returns:
        Optional[DetectionResult]: Detection result containing corner coordinates and visualization,
                                   or None if detection fails
    """
    start_total = time.time()

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Read and validate image
    t0 = time.time()
    image: NDArray[np.uint8] = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    print(f"Image loading took: {time.time() - t0:.3f}s")

    # Convert to grayscale
    t0 = time.time()
    gray: NDArray[np.uint8] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(f"Grayscale conversion took: {time.time() - t0:.3f}s")

    # Apply adaptive thresholding
    t0 = time.time()
    thresh: NDArray[np.uint8] = cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=11,
        C=2,
    )
    print(f"Adaptive thresholding took: {time.time() - t0:.3f}s")

    # Morphological operations
    t0 = time.time()
    kernel: NDArray[np.uint8] = np.ones((3, 3), dtype=np.uint8)
    morph: NDArray[np.uint8] = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    print(f"Morphological operations took: {time.time() - t0:.3f}s")

    # Find contours
    t0 = time.time()
    contours, _ = cv2.findContours(
        morph, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None
    print(f"Contour finding took: {time.time() - t0:.3f}s")

    # Process contours and find corners
    t0 = time.time()
    puzzle_contour: NDArray[np.int32] = max(contours, key=cv2.contourArea)
    epsilon: float = 0.02 * cv2.arcLength(puzzle_contour, closed=True)
    approx: NDArray[np.int32] = cv2.approxPolyDP(puzzle_contour, epsilon, closed=True)

    if len(approx) != 4:
        return None

    # Convert to array of points and sort corners
    corners_array: NDArray[np.int32] = approx.reshape(4, 2)
    sum_coords: NDArray[np.int32] = corners_array.sum(axis=1)
    tl: Corner = Corner(*corners_array[np.argmin(sum_coords)])
    br: Corner = Corner(*corners_array[np.argmax(sum_coords)])
    diff: NDArray[np.int32] = np.diff(corners_array, axis=1).reshape(-1)
    tr: Corner = Corner(*corners_array[np.argmin(diff)])
    bl: Corner = Corner(*corners_array[np.argmax(diff)])
    print(f"Corner processing took: {time.time() - t0:.3f}s")

    # Visualization
    t0 = time.time()
    corner_img: NDArray[np.uint8] = image.copy()
    corner_markers: list[tuple[Corner, tuple[int, int, int]]] = [
        (tl, (0, 0, 255)),  # Red
        (tr, (0, 255, 0)),  # Green
        (bl, (255, 0, 0)),  # Blue
        (br, (255, 255, 0)),  # Cyan
    ]

    for corner, color in corner_markers:
        cv2.circle(
            corner_img, (corner.x, corner.y), radius=10, color=color, thickness=-1
        )

    cv2.polylines(corner_img, [approx], isClosed=True, color=(0, 255, 0), thickness=2)
    print(f"Visualization creation took: {time.time() - t0:.3f}s")

    corners: Corners = {
        "top_left": tl,
        "top_right": tr,
        "bottom_left": bl,
        "bottom_right": br,
    }

    print(f"Total detection time: {time.time() - start_total:.3f}s")
    return DetectionResult(corners=corners, visualization=corner_img)


def get_perspective_transform(
    image: NDArray[np.uint8],
    corners: Corners,
    output_size: Tuple[int, int] = (500, 500),
) -> NDArray[np.uint8]:
    """
    Apply perspective transformation to obtain a top-down view of the grid.

    Args:
        image (NDArray[np.uint8]): Input image.
        corners (Corners): Detected corners of the grid.
        output_size (Tuple[int, int], optional): Desired output image size (width, height). Defaults to (500, 500).

    Returns:
        NDArray[np.uint8]: Warped image after perspective transformation.
    """
    t0 = time.time()
    width, height = output_size

    # Source points
    src_pts: NDArray[np.float32] = np.float32(
        [
            [corners["top_left"].x, corners["top_left"].y],
            [corners["top_right"].x, corners["top_right"].y],
            [corners["bottom_left"].x, corners["bottom_left"].y],
            [corners["bottom_right"].x, corners["bottom_right"].y],
        ]
    )

    # Destination points
    dst_pts: NDArray[np.float32] = np.float32(
        [[0, 0], [width, 0], [0, height], [width, height]]
    )

    # Calculate and apply transform
    matrix: NDArray[np.float32] = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped: NDArray[np.uint8] = cv2.warpPerspective(image, matrix, (width, height))

    print(f"Perspective transform took: {time.time() - t0:.3f}s")
    return cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)


def process_image(image_path: Path) -> Optional[Str8tsGrid]:
    """
    Main function to process a Str8ts puzzle image.

    This function detects grid corners, applies perspective transformation,
    extracts cells, analyzes cell colors, detects numbers, and reconstructs
    the full Str8ts grid.

    Args:
        image_path (Path): Path to the input Str8ts puzzle image.

    Returns:
        Optional[Str8tsGrid]: The reconstructed Str8ts grid, or None if processing fails.
    """
    start_total = time.time()

    result: Optional[DetectionResult] = detect_grid_corners(image_path)

    if result is None:
        print("Failed to detect grid corners")
        return None

    # Save visualization
    t0 = time.time()
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    cv2.imwrite(str(output_dir / "detected_corners.jpg"), result.visualization)
    print(f"Saving corners visualization took: {time.time() - t0:.3f}s")

    # Get and save perspective transformed image
    t0 = time.time()
    image: NDArray[np.uint8] = cv2.imread(str(image_path))
    warped: NDArray[np.uint8] = get_perspective_transform(image, result.corners)
    cv2.imwrite(str(output_dir / "warped_grid.jpg"), warped)
    print(f"Transform and save took: {time.time() - t0:.3f}s")

    t0 = time.time()
    ocr_engine = RapidOCR()
    ocr_result, _ = ocr_engine(warped)
    print(f"OCR took: {time.time() - t0:.3f}s")

    # Extract and process cells
    t0 = time.time()
    cells = extract_grid_cells(warped)

    # Create a copy of warped image for reconstruction
    processed_grid = warped.copy()

    # Process each cell
    grid_data = []
    for i, row in enumerate(cells):
        grid_row = []
        for j, cell_img in enumerate(row):
            # Calculate cell coordinates
            cell_x1 = j * (warped.shape[1] // 9)
            cell_y1 = i * (warped.shape[0] // 9)
            cell_x2 = (j + 1) * (warped.shape[1] // 9)
            cell_y2 = (i + 1) * (warped.shape[0] // 9)

            # Analyze cell color
            color = analyze_cell_color(cell_img)

            if color == CellColor.BLACK:
                # Invert only the black cell
                inverted_cell = cv2.bitwise_not(cell_img)

                # Resize inverted cell to match target dimensions
                target_height = cell_y2 - cell_y1
                target_width = cell_x2 - cell_x1
                inverted_cell_resized = cv2.resize(
                    inverted_cell, (target_width, target_height)
                )

                # Replace the cell in the processed grid
                processed_grid[cell_y1:cell_y2, cell_x1:cell_x2] = inverted_cell_resized

    cv2.imwrite(str(output_dir / "warped_grid_inverted.jpg"), processed_grid)
    ocr_engine = RapidOCR()
    ocr_result_inverted, _ = ocr_engine(processed_grid)
    print(f"OCR inverted took: {time.time() - t0:.3f}s")

    # Process each cell
    grid_data = []
    for i, row in enumerate(cells):
        grid_row = []
        for j, cell_img in enumerate(row):
            color = analyze_cell_color(cell_img)
            # Calculate cell coordinates within the warped grid
            cell_x1 = j * (warped.shape[1] // 9)
            cell_y1 = i * (warped.shape[0] // 9)
            cell_x2 = (j + 1) * (warped.shape[1] // 9)
            cell_y2 = (i + 1) * (warped.shape[0] // 9)

            # Detect digit using OCR results
            detected_number = detect_digit(
                (cell_x1, cell_y1, cell_x2, cell_y2), ocr_result
            )
            if color == CellColor.BLACK and detected_number is None:
                detected_number = detect_digit(
                    (cell_x1, cell_y1, cell_x2, cell_y2), ocr_result_inverted
                )

            grid_row.append(Cell(color=color, number=detected_number))

            # Save individual cells for debugging
            # cv2.imwrite(str(output_dir / f"cell_{i}_{j}.jpg"), cell_img)

        grid_data.append(grid_row)
    print(f"Cell processing took: {time.time() - t0:.3f}s")

    print(f"Total processing time: {time.time() - start_total:.3f}s")

    # Print corner coordinates
    print("\nCorner coordinates:")
    for corner_name, coords in result.corners.items():
        print(f"{corner_name}: ({coords.x}, {coords.y})")

    # Print grid contents
    print("\nGrid contents:")
    for i, row in enumerate(grid_data):
        print(f"Row {i}: ", end="")
        for cell in row:
            if cell.color == CellColor.BLACK:
                print(f"[B{cell.number if cell.number else '_'}]", end=" ")
            else:
                print(f" {cell.number if cell.number else '_'} ", end=" ")
        print()

    return Str8tsGrid(cells=grid_data)


if __name__ == "__main__":
    image_path = Path("samples/sample_6.jpeg")
    grid = process_image(image_path)
