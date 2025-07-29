"""
Computer Vision Module for Pest Detection
Demo version that simulates computer vision functionality.
"""

# Try to import the full version, fall back to demo version
try:
    import cv2
    import torch
    import torchvision.transforms as transforms
    
    # If imports succeed, use the full implementation (original code)
    # This would contain the original PestDetector class
    
    class PestDetector:
        """Full computer vision pipeline for pest identification."""
        
        def __init__(self, model_path=None):
            raise ImportError("Full computer vision dependencies not available. Using demo version.")
    
except ImportError:
    # Fall back to demo version
    from .pest_detector_demo import PestDetector
    import logging
    
    logger = logging.getLogger(__name__)
    logger.info("Using demo pest detector (computer vision libraries not available)")
