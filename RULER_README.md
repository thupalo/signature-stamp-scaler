# Ruler Detection and Scale Measurement Tools

This project provides Python tools for detecting rulers in images and using them to determine the real-world scale for measuring objects. The tools use computer vision techniques to automatically detect rulers and calculate pixels-per-centimeter ratios.

## Features

- **Automatic ruler detection** using edge detection and line detection algorithms
- **Manual ruler selection** as fallback when automatic detection fails
- **Interactive object measurement** by clicking and dragging
- **Scale calculation** with high accuracy
- **Multiple measurement tools** for different use cases

## Installation

1. Make sure you have Python 3.7+ installed
2. Install required packages:

```bash
pip install -r requirements.txt
```

Required packages:
- opencv-python
- numpy
- matplotlib
- pillow
- scipy

## Tools Overview

### 1. `ruler_detection.py` - Core Detection Engine
The main detection algorithm with visualization capabilities.

**Usage:**
```bash
python ruler_detection.py
```

**Features:**
- Automatic ruler detection using Hough line transform
- Edge detection with Canny algorithm
- Scale calculation (pixels per centimeter)
- Visualization of detection results
- Example measurements

### 2. `measure_scale.py` - Simple Command-Line Tool
Quick and easy scale detection for any image.

**Usage:**
```bash
python measure_scale.py <image_path> [ruler_length_cm]
```

**Examples:**
```bash
python measure_scale.py AL1.png
python measure_scale.py AL1.png 15.0
python measure_scale.py butterfly.jpg 10.0
```

**Output:**
- Scale factor (pixels per cm)
- Image resolution (DPI)
- Total image dimensions in cm

### 3. `interactive_ruler.py` - Manual Selection Tool
Interactive tool with GUI for manual ruler selection when automatic detection fails.

**Usage:**
```bash
python interactive_ruler.py
```

**Features:**
- Attempts automatic detection first
- Falls back to manual selection via clicking
- Interactive measurement mode
- Real-time distance calculation

### 4. `object_measure.py` - Object Measurement Tool
Comprehensive tool for measuring specific objects in images.

**Usage:**
```bash
python object_measure.py <image_path> [ruler_length_cm]
```

**Features:**
- Click and drag to measure objects
- Real-time dimension display
- Multiple object measurements
- Export measurements to text file

## How It Works

### 1. Ruler Detection Algorithm

1. **Preprocessing:**
   - Convert image to grayscale
   - Apply Gaussian blur to reduce noise

2. **Edge Detection:**
   - Use Canny edge detection
   - Multiple parameter sets for robustness

3. **Line Detection:**
   - Hough Line Transform to detect straight lines
   - Filter for horizontal/vertical lines (typical ruler orientations)
   - Select the longest suitable line as ruler

4. **Scale Calculation:**
   - Measure detected ruler line length in pixels
   - Divide by known ruler length in centimeters
   - Result: pixels per centimeter ratio

### 2. Object Measurement

Once scale is established:
- Measure any distance in pixels
- Convert to centimeters using the scale factor
- Calculate areas, perimeters, etc.

## Usage Examples

### Example 1: Basic Scale Detection
```python
from ruler_detection import RulerDetector
import numpy as np
from loadimg import load_img

# Load image
image = np.array(load_img("AL1.png"))

# Detect ruler (assuming 15cm ruler)
detector = RulerDetector()
scale = detector.detect_ruler(image, known_ruler_length_cm=15.0)

print(f"Scale: {scale:.2f} pixels per cm")
```

### Example 2: Measure Object Dimensions
```python
# After detecting scale...
object_bbox = (100, 100, 200, 150)  # x, y, width, height in pixels
dimensions = detector.measure_object(image, object_bbox)

if dimensions:
    width_cm, height_cm = dimensions
    print(f"Object: {width_cm:.2f} cm × {height_cm:.2f} cm")
```

### Example 3: Interactive Measurement
```python
from interactive_ruler import InteractiveRulerDetector

detector = InteractiveRulerDetector()
detector.load_and_process_image("AL1.png")

# Try automatic detection
scale = detector.automatic_detection(15.0)

if scale is None:
    # Fall back to manual selection
    scale = detector.manual_ruler_selection(15.0)

# Start interactive measurement
detector.interactive_measurement()
```

## Tips for Better Detection

### Image Quality
- Use high-resolution images
- Ensure good lighting and contrast
- Avoid shadows on the ruler

### Ruler Placement
- Place ruler with clear, unobstructed edges
- Horizontal or vertical orientation works best
- Ensure ruler is straight and not curved

### Common Issues and Solutions

**Problem:** Ruler not detected automatically
**Solution:** Use `interactive_ruler.py` for manual selection

**Problem:** Inaccurate scale detection
**Solution:** 
- Verify the ruler length parameter
- Ensure ruler is clearly visible
- Try different known_ruler_length values

**Problem:** Multiple lines detected
**Solution:** Algorithm selects the longest suitable line, but manual selection may be more accurate

## File Structure

```
├── ruler_detection.py      # Core detection algorithm
├── interactive_ruler.py    # Interactive GUI tool
├── measure_scale.py        # Simple command-line tool
├── object_measure.py       # Object measurement tool
├── loadimg.py             # Image loading utilities
├── requirements.txt       # Python dependencies
├── AL1.png               # Example image with ruler
└── README.md             # This file
```

## API Reference

### RulerDetector Class

**Methods:**
- `detect_ruler(image, known_ruler_length_cm)` - Main detection method
- `measure_object(image, coordinates)` - Measure object dimensions
- `visualize_detection(image)` - Show detection results

**Properties:**
- `pixels_per_cm` - Calculated scale factor
- `ruler_line` - Detected ruler line coordinates

### InteractiveRulerDetector Class

**Methods:**
- `load_and_process_image(path)` - Load image
- `automatic_detection(length)` - Try automatic detection
- `manual_ruler_selection(length)` - Manual selection GUI
- `interactive_measurement()` - Interactive measurement mode

## Contributing

Feel free to improve the detection algorithms or add new features:

1. **Edge Detection:** Experiment with different Canny parameters
2. **Line Detection:** Tune Hough transform parameters
3. **Filtering:** Improve ruler line selection criteria
4. **GUI:** Enhance the interactive interfaces

## License

This project is open source. Feel free to use and modify as needed.

## Examples with AL1.png

The included `AL1.png` image demonstrates the tool's capabilities:
- Contains a ruler for scale reference
- Successfully detected with default parameters
- Scale: ~25.33 pixels per centimeter

Try running:
```bash
python measure_scale.py AL1.png 15.0
```

This should detect the ruler and display the scale information along with a visualization of the detection results.
