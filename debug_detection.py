#!/usr/bin/env python3
"""
Test script to debug the uploaded image detection issue.
"""

from vision.pest_detector_demo import PestDetector
import os
from pathlib import Path

def debug_detection():
    """Debug the detection process for uploaded images."""
    detector = PestDetector()
    
    # Test the actual file
    test_file = 'test_images/aphids_high.jpg'
    print(f"=== Testing File: {test_file} ===")
    print(f"File exists: {os.path.exists(test_file)}")
    
    if os.path.exists(test_file):
        try:
            result = detector.detect(test_file)
            print("\n=== Detection Results ===")
            print(f"Pest Type: {result['pest_type']}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Severity: {result['severity']}")
            print(f"Threshold check (>0.7): {result['confidence'] > 0.7}")
            print(f"Success: {result['confidence'] > 0.7}")
            
            # Test filename parsing
            filename = Path(test_file).stem.lower()
            print(f"\n=== Filename Analysis ===")
            print(f"Filename stem: '{filename}'")
            
            # Test the pattern matching manually
            pest_id = detector._get_pest_from_filename(filename)
            print(f"Detected pest ID from filename: {pest_id}")
            if pest_id is not None:
                pest_info = detector.pest_classes[pest_id]
                print(f"Pest name: {pest_info['name']}")
            
        except Exception as e:
            print(f"Error during detection: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Test file not found!")
        # List available files
        if os.path.exists('test_images'):
            print("Available test images:")
            for file in os.listdir('test_images'):
                print(f"  - {file}")

if __name__ == "__main__":
    debug_detection()
