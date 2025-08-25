import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from loadimg import load_img
from typing import Tuple, Optional
import tkinter as tk
from tkinter import simpledialog


class InteractiveRulerDetector:
    """
    Interactive ruler detection with manual selection fallback.
    """
    
    def __init__(self):
        self.pixels_per_cm = None
        self.ruler_endpoints = None
        self.image = None
        self.fig = None
        self.ax = None
        self.points = []
        
    def load_and_process_image(self, image_path: str):
        """Load and prepare image for processing."""
        self.image = np.array(load_img(image_path))
        print(f"Image loaded: {self.image.shape}")
        return self.image
    
    def manual_ruler_selection(self, known_length_cm: float = 15.0):
        """
        Allow user to manually select ruler endpoints by clicking.
        """
        if self.image is None:
            print("Please load an image first!")
            return None
            
        print("\nManual Ruler Selection Mode")
        print("Instructions:")
        print("1. Click on one end of the ruler")
        print("2. Click on the other end of the ruler")
        print("3. Close the plot window when done")
        
        self.points = []
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.imshow(self.image)
        self.ax.set_title("Click on ruler endpoints (2 clicks needed)")
        
        # Connect click event
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        
        plt.show()
        
        if len(self.points) >= 2:
            # Calculate distance and scale
            p1, p2 = self.points[0], self.points[1]
            distance_pixels = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            self.pixels_per_cm = distance_pixels / known_length_cm
            self.ruler_endpoints = (p1, p2)
            
            print(f"✓ Manual ruler selection completed!")
            print(f"Distance: {distance_pixels:.2f} pixels")
            print(f"Scale: {self.pixels_per_cm:.2f} pixels/cm")
            
            return self.pixels_per_cm
        else:
            print("Not enough points selected!")
            return None
    
    def _on_click(self, event):
        """Handle mouse click events for manual selection."""
        if event.inaxes != self.ax:
            return
            
        if len(self.points) < 2:
            x, y = event.xdata, event.ydata
            self.points.append((x, y))
            
            # Plot the point
            self.ax.plot(x, y, 'ro', markersize=8)
            
            if len(self.points) == 1:
                self.ax.set_title("Click on the other end of the ruler")
            elif len(self.points) == 2:
                # Draw line between points
                x1, y1 = self.points[0]
                x2, y2 = self.points[1]
                self.ax.plot([x1, x2], [y1, y2], 'r-', linewidth=3)
                self.ax.set_title("Ruler selected! Close this window to continue.")
            
            self.fig.canvas.draw()
    
    def automatic_detection(self, known_length_cm: float = 15.0) -> Optional[float]:
        """
        Attempt automatic ruler detection.
        """
        if self.image is None:
            print("Please load an image first!")
            return None
            
        # Convert to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        
        # Edge detection with multiple parameter sets
        edges1 = cv2.Canny(gray, 50, 150, apertureSize=3)
        edges2 = cv2.Canny(gray, 30, 100, apertureSize=3)
        
        # Combine edge images
        edges = cv2.bitwise_or(edges1, edges2)
        
        # Detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, 
                               minLineLength=50, maxLineGap=10)
        
        if lines is None:
            print("No lines detected automatically.")
            return None
        
        # Find the best ruler candidate
        best_line = self._find_best_ruler_line(lines)
        
        if best_line is None:
            print("No suitable ruler line found automatically.")
            return None
        
        x1, y1, x2, y2 = best_line
        distance_pixels = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        self.pixels_per_cm = distance_pixels / known_length_cm
        self.ruler_endpoints = ((x1, y1), (x2, y2))
        
        print(f"✓ Automatic ruler detection successful!")
        print(f"Scale: {self.pixels_per_cm:.2f} pixels/cm")
        
        return self.pixels_per_cm
    
    def _find_best_ruler_line(self, lines: np.ndarray) -> Optional[np.ndarray]:
        """Find the best ruler line candidate."""
        candidates = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # Calculate angle
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            
            # Prefer horizontal or vertical lines
            angle_score = 0
            if abs(angle) < 10 or abs(angle) > 170:  # Nearly horizontal
                angle_score = 1
            elif abs(abs(angle) - 90) < 10:  # Nearly vertical
                angle_score = 1
            
            # Score based on length and orientation
            score = length * angle_score
            
            if score > 0:
                candidates.append((score, line[0]))
        
        if not candidates:
            return None
        
        # Return the highest scoring line
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    
    def measure_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> Optional[float]:
        """
        Measure distance between two points in centimeters.
        """
        if self.pixels_per_cm is None:
            print("No scale established. Please detect ruler first!")
            return None
        
        distance_pixels = np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
        distance_cm = distance_pixels / self.pixels_per_cm
        
        return distance_cm
    
    def interactive_measurement(self):
        """
        Allow user to measure distances by clicking points.
        """
        if self.pixels_per_cm is None:
            print("No scale established. Please detect ruler first!")
            return
            
        print("\nInteractive Measurement Mode")
        print("Instructions:")
        print("1. Click on first point")
        print("2. Click on second point")
        print("3. Distance will be calculated and displayed")
        print("4. Close window when done")
        
        self.measurement_points = []
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(self.image)
        ax.set_title("Click two points to measure distance")
        
        # Show ruler if detected
        if self.ruler_endpoints:
            p1, p2 = self.ruler_endpoints
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=3, label='Detected Ruler')
            ax.legend()
        
        def on_measurement_click(event):
            if event.inaxes != ax:
                return
                
            if len(self.measurement_points) < 2:
                x, y = event.xdata, event.ydata
                self.measurement_points.append((x, y))
                
                ax.plot(x, y, 'bo', markersize=8)
                
                if len(self.measurement_points) == 1:
                    ax.set_title("Click on second point")
                elif len(self.measurement_points) == 2:
                    # Calculate and display distance
                    p1, p2 = self.measurement_points
                    distance = self.measure_distance(p1, p2)
                    
                    # Draw line
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', linewidth=2)
                    
                    # Add text
                    mid_x = (p1[0] + p2[0]) / 2
                    mid_y = (p1[1] + p2[1]) / 2
                    ax.text(mid_x, mid_y, f'{distance:.2f} cm', 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                           fontsize=12, ha='center')
                    
                    ax.set_title(f"Distance: {distance:.2f} cm (Close window or click to measure again)")
                    
                    print(f"Measured distance: {distance:.2f} cm")
                    
                    # Reset for next measurement
                    self.measurement_points = []
                
                fig.canvas.draw()
        
        fig.canvas.mpl_connect('button_press_event', on_measurement_click)
        plt.show()
    
    def visualize_results(self):
        """Show the image with ruler detection results."""
        if self.image is None:
            print("No image loaded!")
            return
            
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(self.image)
        
        if self.ruler_endpoints and self.pixels_per_cm:
            p1, p2 = self.ruler_endpoints
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=4, label='Detected Ruler')
            
            # Add scale information
            ax.text(10, 30, f'Scale: {self.pixels_per_cm:.2f} pixels/cm', 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
                   fontsize=14, fontweight='bold')
            
            ax.legend()
            ax.set_title("Ruler Detection Results")
        else:
            ax.set_title("Image - No Ruler Detected")
            
        ax.axis('off')
        plt.tight_layout()
        plt.show()


def main():
    """Main function demonstrating the interactive ruler detector."""
    print("Interactive Ruler Detection and Measurement Tool")
    print("=" * 55)
    
    # Initialize detector
    detector = InteractiveRulerDetector()
    
    # Load image
    image_path = "AL1.png"
    try:
        detector.load_and_process_image(image_path)
        print(f"✓ Image loaded: {image_path}")
    except Exception as e:
        print(f"✗ Error loading image: {e}")
        return
    
    # Get ruler length from user
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    
    ruler_length = simpledialog.askfloat(
        "Ruler Length", 
        "Enter the length of your ruler in centimeters:",
        initialvalue=15.0,
        minvalue=1.0,
        maxvalue=100.0
    )
    
    if ruler_length is None:
        ruler_length = 15.0  # Default value
    
    root.destroy()
    
    print(f"Using ruler length: {ruler_length} cm")
    
    # Try automatic detection first
    print("\n1. Attempting automatic ruler detection...")
    scale = detector.automatic_detection(ruler_length)
    
    if scale is None:
        print("\n2. Automatic detection failed. Switching to manual selection...")
        scale = detector.manual_ruler_selection(ruler_length)
    
    if scale is not None:
        # Show results
        detector.visualize_results()
        
        # Start interactive measurement
        print(f"\n3. Scale established: {scale:.2f} pixels/cm")
        print("Starting interactive measurement mode...")
        detector.interactive_measurement()
    else:
        print("\n✗ Could not establish scale. Please try again with a clearer ruler image.")


if __name__ == "__main__":
    main()
