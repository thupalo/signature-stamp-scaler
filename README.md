# Signature Stamp Scaler

A Python application for processing and scaling signatures/stamps in images using computer vision techniques.

## Features

- Image processing with ruler detection for scaling
- Gradio web interface for easy usage
- Support for various image formats
- Debug mode for troubleshooting

## Requirements

- Python 3.11+
- See `requirements.txt` for full dependency list

## Installation

1. Clone the repository:
```bash
git clone https://github.com/thupalo/signature-stamp-scaler.git
cd signature-stamp-scaler
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Web Interface (Gradio)
```bash
python ruler_gradio.py
```

### Command Line
The command line interface processes images by detecting rulers, cropping the image, and saving it with the correct scale.

**Basic Usage:**
```bash
python ruler_01.py input_image.jpg output_image.png
```

**Command Line Arguments:**
- `input_path` - Path to input image (JPEG or PNG format)
- `output_path` - Path to output PNG file (will auto-append .png if not specified)

**Options:**
- `-d, --debug` - Enable debug output for troubleshooting
- `--overwrite` - Overwrite output file if it exists without prompting

**Examples:**
```bash
# Basic usage
python ruler_01.py examples/image.jpg processed_output.png

# With debug mode enabled
python ruler_01.py examples/image.jpg output.png --debug

# Overwrite existing file without prompt
python ruler_01.py input.jpg output.png --overwrite
```

**Supported Input Formats:** `.jpg`, `.jpeg`, `.png`  
**Output Format:** `.png` (automatically enforced)

## Dependencies

- Pillow - Image processing
- OpenCV - Computer vision operations
- NumPy - Numerical operations
- Matplotlib - Plotting and visualization
- Gradio - Web interface
- Requests - HTTP requests

## Project Structure

- `ruler_01.py` - Main processing module
- `ruler_gradio.py` - Gradio web interface
- `requirements.txt` - Python dependencies
- `examples/` - Example images for testing

## License

This project is available under the MIT License.
