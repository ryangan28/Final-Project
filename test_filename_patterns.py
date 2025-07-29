#!/usr/bin/env python3
"""
Test various filename scenarios to ensure robust detection.
"""

from vision.pest_detector_demo import PestDetector

def test_various_filename_scenarios():
    """Test detection with various filename patterns."""
    detector = PestDetector()
    
    test_cases = [
        # Original test case
        ("aphids_high.jpg", "Aphids"),
        ("pest_upload_aphids_high.jpg", "Aphids"),
        # Different naming patterns
        ("aphid_damage.jpg", "Aphids"),
        ("my_aphids_photo.jpg", "Aphids"),
        ("caterpillar_on_leaf.jpg", "Caterpillars"),
        ("spider_mite_problem.jpg", "Spider Mites"),
        ("whitefly_infestation.jpg", "Whitefly"),
        ("thrips_damage.jpg", "Thrips"),
        ("colorado_beetle.jpg", "Colorado Potato Beetle"),
        ("cucumber_beetle_adult.jpg", "Cucumber Beetle"),
        ("flea_beetle_holes.jpg", "Flea Beetle"),
        # Cases that should return random results
        ("random_image.jpg", None),
        ("tmpXYZ123.jpg", None),
    ]
    
    print("=== Testing Various Filename Scenarios ===")
    
    all_passed = True
    for filename, expected_pest in test_cases:
        try:
            pest_id = detector._get_pest_from_filename(filename)
            detected_pest = detector.pest_classes[pest_id]['name'] if pest_id is not None else "None"
            
            if expected_pest is None:
                # For random filenames, any result is acceptable
                status = "✅ PASS"
            else:
                status = "✅ PASS" if detected_pest == expected_pest else "❌ FAIL"
                if detected_pest != expected_pest:
                    all_passed = False
            
            print(f"{filename:25} -> {detected_pest:20} {status}")
            
        except Exception as e:
            print(f"{filename:25} -> ERROR: {e}")
            all_passed = False
    
    return all_passed

if __name__ == "__main__":
    success = test_various_filename_scenarios()
    print(f"\n=== Overall Test {'PASSED' if success else 'FAILED'} ===")
    
    if success:
        print("✅ Filename detection is working correctly for various patterns!")
    else:
        print("❌ Some filename patterns failed - may need adjustment.")
