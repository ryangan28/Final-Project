#!/usr/bin/env python3
"""
Test script to verify the pest detection fixes.
"""

from vision.pest_detector_demo import PestDetector

def test_aphids_detection():
    """Test aphids detection with the high quality image."""
    detector = PestDetector()
    
    # Test with aphids filename
    result = detector.detect('test_images/aphids_high.jpg')
    
    print('=== Test Results ===')
    print(f'Pest Type: {result["pest_type"]}')
    print(f'Confidence: {result["confidence"]:.3f}')
    print(f'Severity: {result["severity"]}')
    print(f'Success: {result["confidence"] > 0.7}')
    
    return result["confidence"] > 0.7

if __name__ == "__main__":
    success = test_aphids_detection()
    print(f'\n=== Test {"PASSED" if success else "FAILED"} ===')
