#!/usr/bin/env python3
"""
Ruler Detection Demo - Showcase all capabilities

This script demonstrates the complete ruler detection and measurement pipeline.
"""

import sys
import os
from ruler_detection import RulerDetector
from loadimg import load_img
import numpy as np
import matplotlib.pyplot as plt


def demo_complete_pipeline(image_path: str = "AL1.png", ruler_length: float = 15.0):
    """Complete demonstration of ruler detection capabilities."""
    
    print("🎯 RULER DETECTION COMPLETE DEMO")
    print("=" * 50)
    
    # Step 1: Load and examine image
    print("📷 STEP 1: Loading Image")
    print("-" * 25)
    
    try:
        pil_image = load_img(image_path)
        image_array = np.array(pil_image)
        print(f"✅ Loaded: {image_path}")
        print(f"📐 Dimensions: {image_array.shape[1]} × {image_array.shape[0]} pixels")
        print(f"🎨 Channels: {image_array.shape[2] if len(image_array.shape) > 2 else 1}")
        print(f"💾 File size: {os.path.getsize(image_path) / 1024:.1f} KB")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return
    
    # Step 2: Ruler Detection
    print(f"\n🔍 STEP 2: Ruler Detection")
    print("-" * 25)
    print(f"🎯 Target ruler length: {ruler_length} cm")
    
    detector = RulerDetector()
    scale = detector.detect_ruler(image_array, ruler_length)
    
    if scale is not None:
        print(f"✅ Ruler detected successfully!")
        print(f"📊 Scale: {scale:.3f} pixels per centimeter")
        print(f"📏 Resolution: {scale * 2.54:.1f} pixels per inch")
        print(f"🔬 Precision: {1/scale:.3f} cm per pixel")
    else:
        print(f"❌ Ruler detection failed")
        return
    
    # Step 3: Scale Analysis
    print(f"\n📊 STEP 3: Scale Analysis")
    print("-" * 25)
    
    total_width_cm = image_array.shape[1] / scale
    total_height_cm = image_array.shape[0] / scale
    total_area_cm2 = total_width_cm * total_height_cm
    
    print(f"🌐 Full image measurements:")
    print(f"   Width: {total_width_cm:.1f} cm")
    print(f"   Height: {total_height_cm:.1f} cm") 
    print(f"   Area: {total_area_cm2:.1f} cm²")
    print(f"   Diagonal: {np.sqrt(total_width_cm**2 + total_height_cm**2):.1f} cm")
    
    # Step 4: Sample Measurements
    print(f"\n📏 STEP 4: Sample Measurements")
    print("-" * 25)
    
    # Define some sample areas to measure
    sample_areas = [
        ("Top-left quarter", (0, 0, image_array.shape[1]//2, image_array.shape[0]//2)),
        ("Center square", (image_array.shape[1]//4, image_array.shape[0]//4, 
                          image_array.shape[1]//2, image_array.shape[0]//2)),
        ("Bottom-right quarter", (image_array.shape[1]//2, image_array.shape[0]//2,
                                 image_array.shape[1]//2, image_array.shape[0]//2))
    ]
    
    for name, (x, y, w, h) in sample_areas:
        dimensions = detector.measure_object(image_array, (x, y, x+w, y+h))
        if dimensions:
            width_cm, height_cm = dimensions
            area_cm2 = width_cm * height_cm
            print(f"📐 {name}:")
            print(f"   Size: {width_cm:.1f} × {height_cm:.1f} cm")
            print(f"   Area: {area_cm2:.1f} cm²")
    
    # Step 5: Accuracy Assessment
    print(f"\n🎯 STEP 5: Accuracy Assessment")
    print("-" * 25)
    
    if detector.ruler_line is not None:
        x1, y1, x2, y2 = detector.ruler_line
        detected_length_pixels = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        detected_length_cm = detected_length_pixels / scale
        
        error_cm = abs(detected_length_cm - ruler_length)
        error_percent = (error_cm / ruler_length) * 100
        
        print(f"🎯 Detection accuracy:")
        print(f"   Expected: {ruler_length:.2f} cm")
        print(f"   Detected: {detected_length_cm:.2f} cm")
        print(f"   Error: {error_cm:.3f} cm ({error_percent:.2f}%)")
        
        if error_percent < 5:
            print(f"✅ Excellent accuracy!")
        elif error_percent < 10:
            print(f"✅ Good accuracy!")
        else:
            print(f"⚠️  Consider rechecking ruler length parameter")
    
    # Step 6: Visualization
    print(f"\n🎨 STEP 6: Creating Visualization")
    print("-" * 25)
    
    try:
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Ruler Detection Analysis - {os.path.basename(image_path)}', fontsize=16, fontweight='bold')
        
        # Original image
        axes[0,0].imshow(image_array)
        axes[0,0].set_title('Original Image')
        axes[0,0].axis('off')
        
        # Image with ruler detection
        result_image = image_array.copy()
        if detector.ruler_line is not None:
            x1, y1, x2, y2 = detector.ruler_line
            # Draw ruler line in red
            import cv2
            cv2.line(result_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 6)
            # Add scale text
            cv2.putText(result_image, f"Scale: {scale:.2f} px/cm", 
                       (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
        
        axes[0,1].imshow(result_image)
        axes[0,1].set_title('Detected Ruler')
        axes[0,1].axis('off')
        
        # Scale information plot
        axes[1,0].text(0.1, 0.9, f"Scale Factor: {scale:.3f} px/cm", fontsize=14, fontweight='bold')
        axes[1,0].text(0.1, 0.8, f"Resolution: {scale*2.54:.1f} DPI", fontsize=12)
        axes[1,0].text(0.1, 0.7, f"Image Width: {total_width_cm:.1f} cm", fontsize=12)
        axes[1,0].text(0.1, 0.6, f"Image Height: {total_height_cm:.1f} cm", fontsize=12)
        axes[1,0].text(0.1, 0.5, f"Total Area: {total_area_cm2:.1f} cm²", fontsize=12)
        if detector.ruler_line is not None:
            axes[1,0].text(0.1, 0.4, f"Ruler Length: {detected_length_cm:.2f} cm", fontsize=12)
            axes[1,0].text(0.1, 0.3, f"Detection Error: {error_percent:.2f}%", fontsize=12)
        axes[1,0].set_xlim(0, 1)
        axes[1,0].set_ylim(0, 1)
        axes[1,0].set_title('Measurement Summary')
        axes[1,0].axis('off')
        
        # Sample measurements visualization
        sample_image = image_array.copy()
        colors = [(255, 255, 0), (0, 255, 255), (255, 0, 255)]  # Yellow, Cyan, Magenta
        
        for i, (name, (x, y, w, h)) in enumerate(sample_areas):
            color = colors[i % len(colors)]
            cv2.rectangle(sample_image, (x, y), (x+w, y+h), color, 4)
            cv2.putText(sample_image, f"{i+1}", (x+10, y+30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        axes[1,1].imshow(sample_image)
        axes[1,1].set_title('Sample Measurements')
        axes[1,1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print(f"✅ Visualization complete!")
        
    except Exception as e:
        print(f"⚠️  Visualization error: {e}")
    
    # Step 7: Summary
    print(f"\n📋 STEP 7: Summary")
    print("-" * 25)
    print(f"🎯 Detection: {'✅ SUCCESS' if scale else '❌ FAILED'}")
    print(f"📊 Scale: {scale:.3f} pixels/cm")
    print(f"🎨 Image: {total_width_cm:.1f} × {total_height_cm:.1f} cm")
    print(f"🔍 Accuracy: {100-error_percent:.1f}%")
    
    return {
        'success': scale is not None,
        'scale': scale,
        'image_dimensions_cm': (total_width_cm, total_height_cm),
        'accuracy_percent': 100 - error_percent if detector.ruler_line else None,
        'measurements': sample_areas
    }


def main():
    """Main demo function."""
    print("🚀 RULER DETECTION TOOLKIT DEMO")
    print("=" * 60)
    print("This demo showcases all capabilities of the ruler detection system.")
    print("")
    
    # Get parameters
    image_path = sys.argv[1] if len(sys.argv) > 1 else "AL1.png"
    ruler_length = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0
    
    print(f"🎯 Target Image: {image_path}")
    print(f"📏 Ruler Length: {ruler_length} cm")
    print("")
    
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file '{image_path}' not found!")
        print("Usage: python demo.py [image_path] [ruler_length_cm]")
        return
    
    # Run complete demo
    result = demo_complete_pipeline(image_path, ruler_length)
    
    print("\n" + "=" * 60)
    if result['success']:
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print(f"Your image scale: {result['scale']:.3f} pixels per centimeter")
        print("\nNext steps:")
        print("- Use measure_scale.py for quick measurements")
        print("- Use interactive_ruler.py for manual selection")
        print("- Use object_measure.py for detailed object analysis")
    else:
        print("❌ DEMO FAILED - RULER NOT DETECTED")
        print("Try:")
        print("- Checking the ruler length parameter")
        print("- Using interactive_ruler.py for manual selection")
        print("- Ensuring the ruler is clearly visible")


if __name__ == "__main__":
    main()
