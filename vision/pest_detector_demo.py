"""
Simplified Pest Detector for Demo Purposes
Simulates computer vision functionality without requiring heavy ML libraries.
"""

import logging
import random
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class PestDetector:
    """Simplified pest detector that simulates computer vision functionality."""
    
    def __init__(self, model_path=None):
        """
        Initialize the pest detector.
        
        Args:
            model_path (str): Path to the pre-trained model (unused in demo)
        """
        logger.info("Initializing demo pest detector")
        
        # Load pest classes
        self.pest_classes = self._load_pest_classes()
        logger.info("Demo pest detection model loaded successfully")
        
    def _load_pest_classes(self):
        """Load pest class definitions."""
        # Common agricultural pests for organic farming
        pest_classes = {
            0: {
                'name': 'Aphids',
                'scientific_name': 'Aphidoidea',
                'severity_indicators': ['colony_size', 'leaf_damage'],
                'crops_affected': ['tomatoes', 'peppers', 'lettuce', 'beans']
            },
            1: {
                'name': 'Caterpillars',
                'scientific_name': 'Lepidoptera larvae',
                'severity_indicators': ['leaf_holes', 'frass_presence'],
                'crops_affected': ['cabbage', 'broccoli', 'tomatoes', 'corn']
            },
            2: {
                'name': 'Spider Mites',
                'scientific_name': 'Tetranychidae',
                'severity_indicators': ['stippling', 'webbing'],
                'crops_affected': ['beans', 'tomatoes', 'strawberries']
            },
            3: {
                'name': 'Whitefly',
                'scientific_name': 'Aleyrodidae',
                'severity_indicators': ['yellowing', 'sticky_honeydew'],
                'crops_affected': ['tomatoes', 'peppers', 'cucumbers']
            },
            4: {
                'name': 'Thrips',
                'scientific_name': 'Thysanoptera',
                'severity_indicators': ['silver_streaks', 'black_specks'],
                'crops_affected': ['onions', 'tomatoes', 'peppers']
            },
            5: {
                'name': 'Colorado Potato Beetle',
                'scientific_name': 'Leptinotarsa decemlineata',
                'severity_indicators': ['defoliation', 'orange_eggs'],
                'crops_affected': ['potatoes', 'tomatoes', 'eggplants']
            },
            6: {
                'name': 'Cucumber Beetle',
                'scientific_name': 'Diabrotica',
                'severity_indicators': ['bacterial_wilt', 'feeding_damage'],
                'crops_affected': ['cucumbers', 'squash', 'melons']
            },
            7: {
                'name': 'Flea Beetle',
                'scientific_name': 'Chrysomelidae',
                'severity_indicators': ['shot_holes', 'seedling_damage'],
                'crops_affected': ['eggplants', 'potatoes', 'radishes']
            }
        }
        return pest_classes
    
    def preprocess_image(self, image_path):
        """
        Simulate image preprocessing.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: Simulated image data
        """
        try:
            # Simulate image loading and preprocessing
            logger.info(f"Preprocessing image: {image_path}")
            
            # Check if file exists
            if isinstance(image_path, str) and not Path(image_path).exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Simulate preprocessing results
            return {
                'processed': True,
                'dimensions': (224, 224, 3),
                'format': 'RGB'
            }
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise
    
    def analyze_severity(self, image_path, pest_type):
        """
        Simulate pest infestation severity analysis.
        
        Args:
            image_path (str): Path to the image
            pest_type (str): Identified pest type
            
        Returns:
            str: Severity level (low, medium, high)
        """
        try:
            # Simulate severity analysis based on file name or random
            filename = Path(image_path).stem.lower() if isinstance(image_path, str) else ""
            
            # Use filename hints if available
            if 'severe' in filename or 'high' in filename:
                return 'high'
            elif 'mild' in filename or 'low' in filename:
                return 'low'
            elif 'medium' in filename:
                return 'medium'
            else:
                # Random severity for demo
                return random.choice(['low', 'medium', 'high'])
                
        except Exception as e:
            logger.warning(f"Could not analyze severity: {str(e)}")
            return 'medium'  # Default to medium if analysis fails
    
    def detect(self, image_path):
        """
        Simulate pest detection and identification.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: Detection results including pest type, confidence, and severity
        """
        try:
            logger.info(f"Analyzing image: {image_path}")
            
            # Preprocess image (simulation)
            self.preprocess_image(image_path)
            
            # Simulate model inference
            # Use filename hints if available for demo purposes
            filename = Path(image_path).stem.lower() if isinstance(image_path, str) else ""
            
            # Determine pest type based on filename or random
            pest_id = None
            for pid, info in self.pest_classes.items():
                if info['name'].lower() in filename:
                    pest_id = pid
                    break
            
            if pest_id is None:
                # Random pest for demo
                pest_id = random.randint(0, len(self.pest_classes) - 1)
            
            # Generate confidence (higher for specific filenames)
            if any(pest['name'].lower() in filename for pest in self.pest_classes.values()):
                confidence_score = random.uniform(0.85, 0.95)
            else:
                confidence_score = random.uniform(0.75, 0.90)
            
            # Get pest information
            pest_info = self.pest_classes[pest_id]
            
            # Analyze severity
            severity = self.analyze_severity(image_path, pest_info['name'])
            
            results = {
                'pest_type': pest_info['name'],
                'scientific_name': pest_info['scientific_name'],
                'confidence': confidence_score,
                'severity': severity,
                'crops_affected': pest_info['crops_affected'],
                'pest_id': pest_id
            }
            
            logger.info(f"Detected: {pest_info['name']} (confidence: {confidence_score:.3f})")
            return results
            
        except Exception as e:
            logger.error(f"Error in pest detection: {str(e)}")
            raise
    
    def batch_detect(self, image_paths):
        """
        Detect pests in multiple images.
        
        Args:
            image_paths (list): List of image paths
            
        Returns:
            list: List of detection results
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.detect(image_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {image_path}: {str(e)}")
                results.append({'error': str(e), 'image_path': image_path})
        
        return results
