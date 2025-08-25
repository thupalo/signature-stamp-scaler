#!/usr/bin/env python3
"""
Simple Ruler Detection and Scale Measurement Tool

Usage:
    python measure_scale.py [image_path] [ruler_length_cm]

Examples:
    python measure_scale.py AL1.png 15.0
    python measure_scale.py butterfly.jpg 10.0
"""

import sys
import os
from ruler_detection import RulerDetector, detect_and_measure_example
from loadimg import load_img
import numpy as np


def simple_measure(image_path: str, ruler_length_cm: float = 15.0, show_visualization: bool = True):
    """
    Simple measurement function with basic output.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found!")
        return None
    
    try:
        # Load image
        pil_image = load_img(image_path)
        image_array = np.array(pil_image)
        
        print(f"📷 Analyzing: {image_path}")
        print(f"📐 Expected ruler length: {ruler_length_cm} cm")
        print(f"📊 Image size: {image_array.shape[1]} × {image_array.shape[0]} pixels")
        
        # Detect ruler
        detector = RulerDetector()
        scale = detector.detect_ruler(image_array, ruler_length_cm)
        
        if scale is not None:
            print(f"✅ SUCCESS: Ruler detected!")
            print(f"🔍 Scale: {scale:.2f} pixels per centimeter")
            print(f"📏 Resolution: {scale * 2.54:.1f} pixels per inch")
            
            # Calculate some useful measurements
            width_cm = image_array.shape[1] / scale
            height_cm = image_array.shape[0] / scale
            
            print(f"📐 Total image dimensions:")
            print(f"   Width: {width_cm:.1f} cm ({image_array.shape[1]} pixels)")
            print(f"   Height: {height_cm:.1f} cm ({image_array.shape[0]} pixels)")
            
            # Show visualization if requested
            if show_visualization:
                try:
                    detector.visualize_detection(image_array, f"Scale Detection - {os.path.basename(image_path)}")
                except:
                    print("📱 Visualization skipped (display not available)")
            
            return {
                'scale': scale,
                'pixels_per_cm': scale,
                'pixels_per_inch': scale * 2.54,
                'image_width_cm': width_cm,
                'image_height_cm': height_cm,
                'image_width_px': image_array.shape[1],
                'image_height_px': image_array.shape[0]
            }
        else:
            print("❌ FAILED: No ruler detected")
            print("💡 Tips:")
            print("   - Make sure the ruler is clearly visible and straight")
            print("   - Try adjusting the ruler length parameter")
            print("   - Use the interactive version: python interactive_ruler.py")
            return None
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return None


def main():
    """Main command-line interface."""
    print("🔧 Simple Ruler Detection Tool")
    print("=" * 40)
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python measure_scale.py <image_path> [ruler_length_cm]")
        print("\nExamples:")
        print("  python measure_scale.py AL1.png")
        print("  python measure_scale.py AL1.png 15.0")
        print("  python measure_scale.py butterfly.jpg 10.0")
        return
    
    image_path = sys.argv[1]
    ruler_length_cm = 15.0  # default
    
    if len(sys.argv) >= 3:
        try:
            ruler_length_cm = float(sys.argv[2])
        except ValueError:
            print(f"Error: Invalid ruler length '{sys.argv[2]}'. Using default 15.0 cm")
            ruler_length_cm = 15.0
    
    # Run measurement
    result = simple_measure(image_path, ruler_length_cm, show_visualization=True)
    
    if result:
        print("\n" + "=" * 40)
        print("✅ MEASUREMENT COMPLETE")
        print(f"📊 Scale: {result['scale']:.2f} pixels/cm")
        print("Use this scale factor for measuring objects in your image!")
    else:
        print("\n" + "=" * 40)
        print("❌ MEASUREMENT FAILED")
        print("Try the interactive version for manual ruler selection:")
        print("python interactive_ruler.py")


if __name__ == "__main__":
    main()
