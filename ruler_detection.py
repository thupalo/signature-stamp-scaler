import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt
from loadimg import load_img


class RulerDetector:
    """
    A class to detect rulers in images and determine scale based on ruler measurements.
    """
    
    def __init__(self):
        self.pixels_per_cm = None
        self.ruler_line = None
        self.detected_marks = []
    
    def detect_ruler(self, image: np.ndarray, known_ruler_length_cm: float = 15.0) -> Optional[float]:
        """
        Detect ruler in image and calculate pixels per centimeter.
        
        Args:
            image: Input image as numpy array
            known_ruler_length_cm: Known length of the ruler in centimeters
            
        Returns:
            pixels_per_cm: Scale factor (pixels per centimeter) or None if no ruler detected
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        
        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                               minLineLength=100, maxLineGap=10)
        
        if lines is None:
            return None
        
        # Find the longest horizontal or vertical line (likely the ruler)
        longest_line = self._find_longest_line(lines)
        
        if longest_line is None:
            return None
        
        self.ruler_line = longest_line
        line_length_pixels = self._calculate_line_length(longest_line)
        
        # Calculate pixels per centimeter
        self.pixels_per_cm = line_length_pixels / known_ruler_length_cm
        
        # Try to detect ruler marks for more accurate scaling
        self._detect_ruler_marks(gray, longest_line)
        
        return self.pixels_per_cm
    
    def _find_longest_line(self, lines: np.ndarray) -> Optional[np.ndarray]:
        """Find the longest line from detected lines."""
        max_length = 0
        longest_line = None
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # Prefer horizontal or vertical lines (ruler orientation)
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            if abs(angle) < 15 or abs(angle) > 165 or abs(abs(angle) - 90) < 15:
                if length > max_length:
                    max_length = length
                    longest_line = line[0]
        
        return longest_line
    
    def _calculate_line_length(self, line: np.ndarray) -> float:
        """Calculate the length of a line in pixels."""
        x1, y1, x2, y2 = line
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def _detect_ruler_marks(self, gray_image: np.ndarray, ruler_line: np.ndarray):
        """
        Detect tick marks on the ruler for more accurate scaling.
        """
        x1, y1, x2, y2 = ruler_line
        
        # Create a region of interest around the ruler line
        roi_width = 50
        
        # Determine if line is more horizontal or vertical
        if abs(x2 - x1) > abs(y2 - y1):  # Horizontal line
            # Extract horizontal ROI
            y_min = max(0, min(y1, y2) - roi_width//2)
            y_max = min(gray_image.shape[0], max(y1, y2) + roi_width//2)
            x_min = min(x1, x2)
            x_max = max(x1, x2)
            roi = gray_image[y_min:y_max, x_min:x_max]
        else:  # Vertical line
            # Extract vertical ROI
            x_min = max(0, min(x1, x2) - roi_width//2)
            x_max = min(gray_image.shape[1], max(x1, x2) + roi_width//2)
            y_min = min(y1, y2)
            y_max = max(y1, y2)
            roi = gray_image[y_min:y_max, x_min:x_max]
        
        # Apply edge detection to find tick marks
        edges = cv2.Canny(roi, 30, 100)
        
        # Find contours (potential tick marks)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours based on size and shape
        self.detected_marks = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 10 < area < 1000:  # Filter by area
                self.detected_marks.append(contour)
    
    def measure_object(self, image: np.ndarray, object_coordinates: Tuple[int, int, int, int]) -> Optional[Tuple[float, float]]:
        """
        Measure an object in the image using the detected ruler scale.
        
        Args:
            image: Input image
            object_coordinates: (x1, y1, x2, y2) bounding box of object
            
        Returns:
            (width_cm, height_cm): Object dimensions in centimeters
        """
        if self.pixels_per_cm is None:
            print("No ruler detected. Please run detect_ruler first.")
            return None
        
        x1, y1, x2, y2 = object_coordinates
        width_pixels = abs(x2 - x1)
        height_pixels = abs(y2 - y1)
        
        width_cm = width_pixels / self.pixels_per_cm
        height_cm = height_pixels / self.pixels_per_cm
        
        return width_cm, height_cm
    
    def visualize_detection(self, image: np.ndarray, title: str = "Ruler Detection"):
        """
        Visualize the ruler detection results.
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Image with ruler detection
        result_image = image.copy()
        
        if self.ruler_line is not None:
            x1, y1, x2, y2 = self.ruler_line
            cv2.line(result_image, (x1, y1), (x2, y2), (255, 0, 0), 3)
            
            # Add text with scale information
            if self.pixels_per_cm is not None:
                text = f"Scale: {self.pixels_per_cm:.2f} pixels/cm"
                cv2.putText(result_image, text, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        axes[1].imshow(result_image)
        axes[1].set_title(f"{title} - Detected Ruler")
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()


def detect_and_measure_example(image_path: str, known_ruler_length: float = 15.0):
    """
    Example function demonstrating ruler detection and measurement.
    
    Args:
        image_path: Path to image file
        known_ruler_length: Known length of ruler in centimeters
    """
    try:
        # Load image
        pil_image = load_img(image_path)
        image_array = np.array(pil_image)
        
        print(f"Loaded image: {image_path}")
        print(f"Image dimensions: {image_array.shape}")
        
        # Initialize detector
        detector = RulerDetector()
        
        # Detect ruler
        scale = detector.detect_ruler(image_array, known_ruler_length)
        
        if scale is not None:
            print(f"✓ Ruler detected!")
            print(f"Scale: {scale:.2f} pixels per centimeter")
            print(f"Image resolution: {1/scale*2.54:.2f} DPI (assuming standard display)")
            
            # Visualize results
            detector.visualize_detection(image_array)
            
            # Example measurement - you can modify these coordinates
            # Let's assume there's an object in the center quarter of the image
            h, w = image_array.shape[:2]
            example_object = (w//4, h//4, 3*w//4, 3*h//4)
            
            dimensions = detector.measure_object(image_array, example_object)
            if dimensions:
                width_cm, height_cm = dimensions
                print(f"\nExample measurement (center area):")
                print(f"Width: {width_cm:.2f} cm")
                print(f"Height: {height_cm:.2f} cm")
                
        else:
            print("✗ No ruler detected in the image.")
            print("Tips for better detection:")
            print("- Ensure the ruler is clearly visible")
            print("- Make sure the ruler edge is straight and unobstructed")
            print("- Try adjusting the known_ruler_length parameter")
            
            # Show original image anyway
            plt.figure(figsize=(10, 8))
            plt.imshow(image_array)
            plt.title("Original Image - No Ruler Detected")
            plt.axis('off')
            plt.show()
            
    except Exception as e:
        print(f"Error processing image: {str(e)}")


if __name__ == "__main__":
    # Example usage with AL1.png
    image_path = "AL1.png"
    
    print("Ruler Detection and Scale Measurement Program")
    print("=" * 50)
    
    # Run detection with default 15cm ruler length
    # Adjust this value based on your actual ruler length
    detect_and_measure_example(image_path, known_ruler_length=15.0)
    
    print("\n" + "=" * 50)
    print("Program completed!")
