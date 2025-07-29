"""
Computer Vision Module for Pest Detection
Demo version that simulates computer vision functionality.
"""

# Try to import the full version, fall back to demo version
try:
    # We only need torch for the full implementation, not cv2
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    
    # If imports succeed, use the full implementation (original code)
    # This would contain the original PestDetector class
    
    class PestDetector:
        """Full computer vision pipeline for pest identification."""
        
        def __init__(self, model_path=None):
            # For now, still use demo implementation since we don't have a trained model
            # In production, this would load the actual trained model
            from .pest_detector_demo import PestDetector as DemoPestDetector
            self._demo = DemoPestDetector(model_path)
        
        def __getattr__(self, name):
            # Delegate all method calls to the demo implementation
            return getattr(self._demo, name)
    
except ImportError:
    # Fall back to demo version
    from .pest_detector_demo import PestDetector
    import logging
    
    logger = logging.getLogger(__name__)
    logger.info("Using demo pest detector (computer vision libraries not available)")
