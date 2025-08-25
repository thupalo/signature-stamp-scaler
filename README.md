# Parafka - Background Removal Tool

This project uses BiRefNet for AI-powered background removal from images.

## Setup

A virtual environment has been created with the following dependencies installed:
- gradio
- torch
- torchvision 
- transformers
- pillow
- requests

## Usage

To run the application:

```powershell
C:/Users/tadeusz.hupalo/Documents/Projects/parafka/.venv/Scripts/python.exe init.py
```

## Required Files

You'll need to add example images to the project directory:
- `butterfly.jpg` - Referenced as an example image in the code (you can use any image file with this name)

## Features

The application provides three interfaces:
1. **Image Upload** - Upload an image file directly
2. **URL Input** - Process an image from a URL
3. **File Output** - Upload an image and download the processed PNG file

## Model

Uses the BiRefNet model from ZhengPeng7/BiRefNet for background segmentation.
Note: This version is optimized to run on CPU, making it accessible without requiring a GPU.

## Performance Note

Running on CPU will be slower than GPU but doesn't require specialized hardware. Processing times will vary based on your CPU performance and image size.
