from PIL import Image
import requests
from io import BytesIO
from typing import Union
import numpy as np

def load_img(image: Union[str, Image.Image], output_type: str = "pil") -> Image.Image:
    """
    Load an image from various sources (file path, URL, or PIL Image).

    Args:
        image: Can be a file path (str), URL (str), or PIL Image object
        output_type: Output format, currently only supports "pil"

    Returns:
        PIL Image object
    """
    if isinstance(image, Image.Image):
        return image
    elif isinstance(image, np.ndarray):
        return Image.open(BytesIO(image))
    elif isinstance(image, str):
        if image.startswith(('http://', 'https://')):
            # Load from URL
            response = requests.get(image)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        else:
            # Load from file path
            return Image.open(image)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

