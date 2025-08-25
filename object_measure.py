"""
Object Measurement Tool - Measure specific objects using detected ruler scale

This script demonstrates how to:
1. Detect ruler scale in an image
2. Allow user to select objects by clicking
3. Calculate real-world dimensions

Usage: python object_measure.py [image_path] [ruler_length_cm]
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys
import os
from ruler_detection import RulerDetector
from loadimg import load_img


class ObjectMeasurer:
    """Tool for measuring objects in images using ruler scale."""
    
    def __init__(self):
        self.detector = RulerDetector()
        self.image = None
        self.scale = None
        self.measurements = []
        
    def load_image_and_detect_scale(self, image_path: str, ruler_length_cm: float = 15.0):
        """Load image and detect ruler scale."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file '{image_path}' not found!")
            
        # Load image
        pil_image = load_img(image_path)
        self.image = np.array(pil_image)
        
        print(f"📷 Image loaded: {image_path}")
        print(f"📐 Image size: {self.image.shape[1]} × {self.image.shape[0]} pixels")
        
        # Detect ruler scale
        self.scale = self.detector.detect_ruler(self.image, ruler_length_cm)
        
        if self.scale is not None:
            print(f"✅ Ruler detected! Scale: {self.scale:.2f} pixels/cm")
            return True
        else:
            print("❌ Ruler detection failed!")
            return False
    
    def measure_objects_interactive(self):
        """Interactive object measurement by clicking and dragging."""
        if self.scale is None:
            print("No scale detected. Cannot measure objects.")
            return
            
        print("\n🖱️  Interactive Object Measurement")
        print("Instructions:")
        print("1. Click and drag to draw rectangles around objects")
        print("2. Release mouse to complete measurement")
        print("3. Close window when finished")
        
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.imshow(self.image)
        ax.set_title("Click and drag to measure objects")
        
        # Show ruler detection
        if self.detector.ruler_line is not None:
            x1, y1, x2, y2 = self.detector.ruler_line
            ax.plot([x1, x2], [y1, y2], 'r-', linewidth=3, label='Detected Ruler')
            ax.legend()
        
        # Variables for rectangle drawing
        self.start_point = None
        self.current_rect = None
        self.is_drawing = False
        
        def on_press(event):
            if event.inaxes != ax:
                return
            self.start_point = (event.xdata, event.ydata)
            self.is_drawing = True
            
        def on_motion(event):
            if not self.is_drawing or event.inaxes != ax:
                return
                
            # Remove previous rectangle
            if self.current_rect is not None:
                self.current_rect.remove()
            
            # Draw new rectangle
            x1, y1 = self.start_point
            x2, y2 = event.xdata, event.ydata
            
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            
            self.current_rect = Rectangle(
                (min(x1, x2), min(y1, y2)), width, height,
                linewidth=2, edgecolor='blue', facecolor='none', alpha=0.7
            )
            ax.add_patch(self.current_rect)
            fig.canvas.draw()
            
        def on_release(event):
            if not self.is_drawing or event.inaxes != ax:
                return
                
            self.is_drawing = False
            
            if self.start_point is not None:
                x1, y1 = self.start_point
                x2, y2 = event.xdata, event.ydata
                
                # Calculate dimensions
                width_pixels = abs(x2 - x1)
                height_pixels = abs(y2 - y1)
                
                width_cm = width_pixels / self.scale
                height_cm = height_pixels / self.scale
                
                # Store measurement
                measurement = {
                    'bbox': (min(x1, x2), min(y1, y2), width_pixels, height_pixels),
                    'width_cm': width_cm,
                    'height_cm': height_cm,
                    'area_cm2': width_cm * height_cm
                }
                self.measurements.append(measurement)
                
                # Add text annotation
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                text = f"W: {width_cm:.1f}cm\nH: {height_cm:.1f}cm"
                ax.text(center_x, center_y, text,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
                       fontsize=10, ha='center', va='center')
                
                print(f"📏 Object #{len(self.measurements)}:")
                print(f"   Width: {width_cm:.2f} cm ({width_pixels:.0f} px)")
                print(f"   Height: {height_cm:.2f} cm ({height_pixels:.0f} px)")
                print(f"   Area: {width_cm * height_cm:.2f} cm²")
                
                fig.canvas.draw()
        
        # Connect events
        fig.canvas.mpl_connect('button_press_event', on_press)
        fig.canvas.mpl_connect('motion_notify_event', on_motion)
        fig.canvas.mpl_connect('button_release_event', on_release)
        
        plt.show()
        
        return self.measurements
    
    def measure_predefined_objects(self, object_boxes):
        """
        Measure predefined object bounding boxes.
        
        Args:
            object_boxes: List of (x, y, width, height) tuples in pixels
        """
        if self.scale is None:
            print("No scale detected. Cannot measure objects.")
            return []
            
        measurements = []
        
        for i, (x, y, w, h) in enumerate(object_boxes):
            width_cm = w / self.scale
            height_cm = h / self.scale
            area_cm2 = width_cm * height_cm
            
            measurement = {
                'object_id': i + 1,
                'bbox_pixels': (x, y, w, h),
                'width_cm': width_cm,
                'height_cm': height_cm,
                'area_cm2': area_cm2
            }
            measurements.append(measurement)
            
            print(f"📏 Object {i+1}:")
            print(f"   Position: ({x:.0f}, {y:.0f}) pixels")
            print(f"   Width: {width_cm:.2f} cm ({w:.0f} px)")
            print(f"   Height: {height_cm:.2f} cm ({h:.0f} px)")
            print(f"   Area: {area_cm2:.2f} cm²")
            
        return measurements
    
    def save_measurements(self, filename: str = "measurements.txt"):
        """Save measurements to a text file."""
        if not self.measurements:
            print("No measurements to save.")
            return
            
        with open(filename, 'w') as f:
            f.write("Object Measurements Report\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"Scale: {self.scale:.2f} pixels per centimeter\n\n")
            
            for i, measurement in enumerate(self.measurements):
                f.write(f"Object {i+1}:\n")
                f.write(f"  Width: {measurement['width_cm']:.2f} cm\n")
                f.write(f"  Height: {measurement['height_cm']:.2f} cm\n")
                f.write(f"  Area: {measurement['area_cm2']:.2f} cm²\n\n")
        
        print(f"📄 Measurements saved to {filename}")


def main():
    """Main function for object measurement tool."""
    print("🔍 Object Measurement Tool")
    print("=" * 30)
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python object_measure.py <image_path> [ruler_length_cm]")
        print("\nExample:")
        print("  python object_measure.py AL1.png 15.0")
        return
    
    image_path = sys.argv[1]
    ruler_length_cm = 15.0
    
    if len(sys.argv) >= 3:
        try:
            ruler_length_cm = float(sys.argv[2])
        except ValueError:
            print(f"Invalid ruler length. Using default: {ruler_length_cm} cm")
    
    # Initialize measurer
    measurer = ObjectMeasurer()
    
    try:
        # Load image and detect scale
        if measurer.load_image_and_detect_scale(image_path, ruler_length_cm):
            
            # Start interactive measurement
            print(f"🎯 Scale established: {measurer.scale:.2f} pixels/cm")
            measurements = measurer.measure_objects_interactive()
            
            # Save measurements if any were made
            if measurements:
                measurer.save_measurements("object_measurements.txt")
                print(f"\n📊 Total objects measured: {len(measurements)}")
            else:
                print("\n📊 No objects were measured.")
                
        else:
            print("❌ Could not establish scale. Please check your image and ruler.")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
