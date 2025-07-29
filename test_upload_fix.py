#!/usr/bin/env python3
"""
Test script to verify the filename preservation fix works.
"""

from vision.pest_detector_demo import PestDetector
import os
import tempfile
import shutil

def test_filename_preservation():
    """Test that the filename preservation fix works."""
    detector = PestDetector()
    
    # Simulate the upload process
    original_file = 'test_images/aphids_high.jpg'
    print(f"=== Testing Filename Preservation Fix ===")
    print(f"Original file: {original_file}")
    
    if not os.path.exists(original_file):
        print("❌ Original test file not found!")
        return False
    
    # Simulate what happens in the upload process
    original_filename = "aphids_high.jpg"  # This is what uploaded_file.name would be
    
    # Clean filename (same logic as in the fix)
    safe_filename = "".join(c for c in original_filename if c.isalnum() or c in ('_', '-', '.')).lower()
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"pest_upload_{safe_filename}")
    
    print(f"Temporary path: {temp_path}")
    print(f"Safe filename: {safe_filename}")
    
    try:
        # Copy the original file to simulate the upload save process
        shutil.copy2(original_file, temp_path)
        
        # Test detection with the temporary file
        result = detector.detect(temp_path)
        
        print(f"\n=== Detection Results ===")
        print(f"Pest Type: {result['pest_type']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Severity: {result['severity']}")
        print(f"Success: {result['confidence'] > 0.7}")
        
        # Test filename parsing
        from pathlib import Path
        filename = Path(temp_path).stem.lower()
        print(f"\n=== Filename Analysis ===")
        print(f"Temp filename stem: '{filename}'")
        
        pest_id = detector._get_pest_from_filename(filename)
        print(f"Detected pest ID from filename: {pest_id}")
        
        success = result['confidence'] > 0.7
        return success
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except:
            pass

if __name__ == "__main__":
    success = test_filename_preservation()
    print(f"\n=== Test {'PASSED' if success else 'FAILED'} ===")
    
    if success:
        print("✅ The filename preservation fix should resolve the upload issue!")
    else:
        print("❌ The fix needs further adjustment.")
