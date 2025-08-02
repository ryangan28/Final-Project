"""
Unified Pest Detection Module
Comprehensive computer vision pipeline for pest identification with multiple model backends and graceful degradation.

This module provides a unified interface that combines:
- EfficientNet-B0 ensemble with uncertainty quantification (primary)
- YOLOv8 detection (secondary)
- Basic ML fallback (tertiary)
- Simulation mode (emergency fallback)

Designed for organic farm pest management with offline-first capabilities.
"""

import logging
import random
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

# Try to import ML dependencies with graceful degradation
ML_CAPABILITIES = {}

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    from torchvision import models
    ML_CAPABILITIES['pytorch'] = True
    logger.info("PyTorch loaded successfully")
except ImportError as e:
    logger.warning(f"PyTorch not available: {e}")
    ML_CAPABILITIES['pytorch'] = False

try:
    from PIL import Image, ImageEnhance, ImageFilter
    ML_CAPABILITIES['pil'] = True
    logger.info("PIL loaded successfully")
except ImportError as e:
    logger.warning(f"PIL not available: {e}")
    ML_CAPABILITIES['pil'] = False

try:
    import cv2
    ML_CAPABILITIES['opencv'] = True
    logger.info("OpenCV loaded successfully")
except ImportError as e:
    logger.warning(f"OpenCV not available: {e}")
    ML_CAPABILITIES['opencv'] = False

# YOLO removed - focusing on EfficientNet only
ML_CAPABILITIES['yolo'] = False

# Determine overall ML availability
ML_AVAILABLE = ML_CAPABILITIES['pytorch'] and ML_CAPABILITIES['pil']
logger.info(f"ML capabilities: {ML_CAPABILITIES}")
logger.info(f"Overall ML available: {ML_AVAILABLE}")


class EfficientNetClassifier(nn.Module):
    """EfficientNet-B0 based pest classifier with uncertainty quantification and temperature scaling."""
    
    def __init__(self, num_classes=12, dropout_rate=0.3):
        super(EfficientNetClassifier, self).__init__()
        # Load pre-trained EfficientNet-B0
        self.backbone = models.efficientnet_b0(weights='DEFAULT')
        
        # Replace classifier with custom head to match saved model
        in_features = self.backbone.classifier[1].in_features  # Should be 1280
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),           # 0
            nn.Linear(in_features, 512),        # 1
            nn.ReLU(),                          # 2
            nn.Dropout(dropout_rate),           # 3
            nn.Linear(512, num_classes)         # 4
        )
        
        # Temperature scaling for uncertainty calibration
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        
        # Enable Monte Carlo Dropout for uncertainty estimation
        self.dropout_rate = dropout_rate
    
    def forward(self, x):
        logits = self.backbone(x)
        # Apply temperature scaling
        return logits / self.temperature
    
    def enable_mc_dropout(self):
        """Enable Monte Carlo Dropout for uncertainty estimation."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()


class UnifiedPestDetector:
    """
    Unified pest detection system with multiple backend support and graceful degradation.
    
    Detection Pipeline:
    1. EfficientNet-B0 ensemble (primary) - highest accuracy with uncertainty quantification
    2. YOLOv8 detection (secondary) - fast object detection
    3. Basic ML classifier (tertiary) - simple CNN fallback
    4. Simulation mode (emergency) - for demo/testing without ML
    
    Supports 12 common agricultural pest types for organic farming.
    """
    
    # Comprehensive pest database for organic farming
    PEST_DATABASE = {
        'ants': {
            'scientific_name': 'Formicidae',
            'common_names': ['Garden ants', 'Worker ants', 'Fire ants'],
            'damage_type': 'indirect',
            'severity_indicators': ['mound_size', 'trail_activity', 'plant_damage'],
            'affected_crops': ['strawberries', 'corn', 'young_seedlings'],
            'detection_features': ['segmented_body', 'antennae', 'six_legs', 'colony_formation']
        },
        'bees': {
            'scientific_name': 'Apis mellifera',
            'common_names': ['Honey bees', 'Worker bees', 'Bumble bees'],
            'damage_type': 'beneficial',
            'severity_indicators': ['pollination_activity', 'hive_proximity'],
            'affected_crops': ['fruit_trees', 'vegetables', 'flowering_plants'],
            'detection_features': ['fuzzy_body', 'pollen_baskets', 'wing_pattern', 'flower_visiting']
        },
        'beetle': {
            'scientific_name': 'Coleoptera',
            'common_names': ['Flea beetle', 'Japanese beetle', 'Colorado potato beetle'],
            'damage_type': 'direct',
            'severity_indicators': ['shot_holes', 'defoliation', 'feeding_damage'],
            'affected_crops': ['potatoes', 'tomatoes', 'eggplant', 'brassicas'],
            'detection_features': ['hard_wing_covers', 'chewing_mouthparts', 'metamorphosis_signs']
        },
        'catterpillar': {
            'scientific_name': 'Lepidoptera larvae',
            'common_names': ['Hornworms', 'Cabbage worms', 'Armyworms', 'Loopers'],
            'damage_type': 'direct',
            'severity_indicators': ['chewing_damage', 'frass_presence', 'defoliation_rate'],
            'affected_crops': ['tomatoes', 'cabbage', 'broccoli', 'corn', 'lettuce'],
            'detection_features': ['segmented_body', 'prolegs', 'chewing_damage', 'silk_threads']
        },
        'earthworms': {
            'scientific_name': 'Oligochaeta',
            'common_names': ['Garden earthworms', 'Red worms', 'Night crawlers'],
            'damage_type': 'beneficial',
            'severity_indicators': ['soil_aeration', 'casting_presence'],
            'affected_crops': ['root_vegetables', 'general_soil_health'],
            'detection_features': ['segmented_body', 'moist_appearance', 'soil_tunnels']
        },
        'earwig': {
            'scientific_name': 'Dermaptera',
            'common_names': ['European earwig', 'Common earwig'],
            'damage_type': 'mixed',
            'severity_indicators': ['night_feeding', 'irregular_holes', 'pest_predation'],
            'affected_crops': ['soft_fruits', 'vegetables', 'flowers'],
            'detection_features': ['pincers', 'elongated_body', 'nocturnal_activity']
        },
        'grasshopper': {
            'scientific_name': 'Orthoptera',
            'common_names': ['Locusts', 'Field grasshoppers', 'Migratory grasshoppers'],
            'damage_type': 'direct',
            'severity_indicators': ['defoliation', 'jumping_activity', 'swarm_size'],
            'affected_crops': ['grasses', 'grains', 'vegetables', 'fruit_trees'],
            'detection_features': ['powerful_hind_legs', 'wing_development', 'chewing_mouthparts']
        },
        'moth': {
            'scientific_name': 'Lepidoptera',
            'common_names': ['Codling moth', 'Cabbage moth', 'Corn borer'],
            'damage_type': 'direct',
            'severity_indicators': ['larval_tunnels', 'fruit_damage', 'flight_activity'],
            'affected_crops': ['apples', 'cabbage', 'corn', 'tomatoes'],
            'detection_features': ['scaled_wings', 'feathery_antennae', 'nocturnal_behavior']
        },
        'slug': {
            'scientific_name': 'Gastropoda',
            'common_names': ['Garden slug', 'Gray field slug', 'Leopard slug'],
            'damage_type': 'direct',
            'severity_indicators': ['slime_trails', 'irregular_holes', 'moisture_seeking'],
            'affected_crops': ['lettuce', 'cabbage', 'strawberries', 'young_plants'],
            'detection_features': ['slimy_body', 'tentacles', 'slime_trail', 'shell_absence']
        },
        'snail': {
            'scientific_name': 'Gastropoda',
            'common_names': ['Garden snail', 'Brown garden snail', 'Roman snail'],
            'damage_type': 'direct',
            'severity_indicators': ['shell_presence', 'feeding_holes', 'slime_trails'],
            'affected_crops': ['lettuce', 'cabbage', 'strawberries', 'herbs'],
            'detection_features': ['spiral_shell', 'tentacles', 'slimy_body', 'retractable_head']
        },
        'wasp': {
            'scientific_name': 'Hymenoptera',
            'common_names': ['Paper wasp', 'Yellow jacket', 'Parasitic wasp'],
            'damage_type': 'mixed',
            'severity_indicators': ['nest_proximity', 'predation_activity', 'fruit_damage'],
            'affected_crops': ['fruits', 'beneficial_as_predator'],
            'detection_features': ['narrow_waist', 'smooth_body', 'folded_wings', 'aggressive_behavior']
        },
        'weevil': {
            'scientific_name': 'Curculionidae',
            'common_names': ['Boll weevil', 'Rice weevil', 'Grain weevil'],
            'damage_type': 'direct',
            'severity_indicators': ['puncture_holes', 'grain_damage', 'larval_presence'],
            'affected_crops': ['cotton', 'grains', 'stored_products', 'fruit_trees'],
            'detection_features': ['elongated_snout', 'hard_body', 'small_size', 'boring_holes']
        }
    }
    
    def __init__(self, model_path=None, confidence_threshold=0.4, uncertainty_threshold=0.8):
        """
        Initialize the unified pest detection system.
        
        Args:
            model_path (str, optional): Path to model files directory
            confidence_threshold (float): Minimum confidence for positive detection
            uncertainty_threshold (float): Maximum uncertainty for confident prediction
        """
        self.model_path = Path(model_path) if model_path else Path("models")
        self.confidence_threshold = confidence_threshold
        self.uncertainty_threshold = uncertainty_threshold
        
        # Detection backends
        self.efficientnet_models = []
        self.basic_model = None
        self.class_mapping = {}
        
        # Initialize available backends
        self._initialize_backends()
        
        # Image preprocessing pipeline
        self._setup_transforms()
        
        logger.info(f"UnifiedPestDetector initialized with {len(self._get_available_backends())} backend(s)")
    
    def _initialize_backends(self):
        """Initialize all available detection backends."""
        
        # 1. EfficientNet Ensemble (Primary)
        if ML_CAPABILITIES['pytorch']:
            self._load_efficientnet_ensemble()
        
        # 2. Basic CNN (Fallback)
        if ML_CAPABILITIES['pytorch']:
            self._load_basic_model()
    
    def _load_efficientnet_ensemble(self):
        """Load EfficientNet-B0 ensemble models."""
        try:
            # Load ensemble of 5 models for better accuracy
            ensemble_paths = [
                self.model_path / "improved" / "best_model_fold_0.pth",
                self.model_path / "improved" / "best_model_fold_1.pth",
                self.model_path / "improved" / "best_model_fold_2.pth",
                self.model_path / "improved" / "best_model_fold_3.pth",
                self.model_path / "improved" / "best_model_fold_4.pth"
            ]
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Loading EfficientNet models on device: {device}")
            
            for i, model_path in enumerate(ensemble_paths):
                if model_path.exists():
                    # Load the saved checkpoint
                    checkpoint = torch.load(model_path, map_location=device)
                    
                    # Create model instance
                    model = EfficientNetClassifier(num_classes=12)
                    
                    # Load the model state dict
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model.to(device)
                    model.eval()
                    model.enable_mc_dropout()  # Enable uncertainty estimation
                    
                    self.efficientnet_models.append(model)
                    logger.info(f"Loaded EfficientNet model {i+1}/5 (fold {i}) - Val Acc: {checkpoint.get('val_acc', 'N/A'):.3f}")
            
            if self.efficientnet_models:
                logger.info(f"EfficientNet ensemble loaded with {len(self.efficientnet_models)} models")
                
                # Load class mapping from the first model
                checkpoint = torch.load(ensemble_paths[0], map_location='cpu')
                self.class_mapping = checkpoint.get('class_mapping', {})
                logger.info(f"Loaded class mapping with {self.class_mapping.get('num_classes', 12)} classes")
            else:
                logger.warning("No EfficientNet models found in models/improved/")
                
        except Exception as e:
            logger.error(f"Failed to load EfficientNet ensemble: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _load_basic_model(self):
        """Load basic CNN model as final ML fallback."""
        try:
            basic_path = self.model_path / "basic_pest_classifier.pth"
            if basic_path.exists():
                # Simple CNN for basic classification
                import torchvision.models as models
                self.basic_model = models.resnet18(pretrained=True)
                self.basic_model.fc = torch.nn.Linear(self.basic_model.fc.in_features, 12)
                
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.basic_model.load_state_dict(torch.load(basic_path, map_location=device))
                self.basic_model.to(device)
                self.basic_model.eval()
                logger.info("Basic CNN model loaded successfully")
            else:
                logger.info("Basic model not found - will use simulation if needed")
        except Exception as e:
            logger.error(f"Failed to load basic model: {e}")
    
    def _setup_transforms(self):
        """Setup image preprocessing transforms."""
        # EfficientNet transforms
        self.efficientnet_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Basic transforms
        self.basic_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                               std=[0.5, 0.5, 0.5])
        ])
    
    def _get_available_backends(self):
        """Get list of available detection backends."""
        backends = []
        if self.efficientnet_models:
            backends.append("EfficientNet-B0 Ensemble")
        if self.basic_model:
            backends.append("Basic CNN")
        return backends
    
    def detect_pest(self, image_path):
        """
        Detect pest in image using EfficientNet ensemble.
        
        Args:
            image_path (str or Path): Path to image file
            
        Returns:
            dict: Detection results with confidence and metadata
        """
        try:
            # Load and validate image
            if not ML_CAPABILITIES['pil']:
                return self._create_error_result("PIL not available - cannot process images")
            
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Use EfficientNet ensemble if available
            if self.efficientnet_models:
                return self._detect_with_efficientnet(image)
            elif self.basic_model:
                return self._detect_with_basic(image)
            else:
                return self._create_error_result("No detection models available")
                
        except Exception as e:
            logger.error(f"Error in pest detection: {e}")
            return self._create_error_result(str(e))
    
    def _detect_with_efficientnet(self, image):
        """Detect pest using EfficientNet ensemble with uncertainty quantification."""
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Preprocess image
            input_tensor = self.efficientnet_transform(image).unsqueeze(0).to(device)
            
            # Monte Carlo predictions for uncertainty estimation
            all_predictions = []
            n_samples = 20  # Number of MC samples
            
            with torch.no_grad():
                for model in self.efficientnet_models:
                    model_predictions = []
                    for _ in range(n_samples):
                        model.enable_mc_dropout()  # Enable dropout during inference
                        output = model(input_tensor)
                        probs = F.softmax(output, dim=1)
                        model_predictions.append(probs.cpu().numpy())
                    all_predictions.extend(model_predictions)
            
            # Calculate ensemble statistics
            predictions_array = np.array(all_predictions)
            mean_probs = np.mean(predictions_array, axis=0)[0]
            std_probs = np.std(predictions_array, axis=0)[0]
            
            # Calculate uncertainty metrics
            max_confidence = np.max(mean_probs)
            predicted_class = np.argmax(mean_probs)
            uncertainty = np.mean(std_probs)  # Average uncertainty across classes
            
            # Get class names from loaded mapping or fallback to database
            if self.class_mapping and 'classes' in self.class_mapping:
                class_names = self.class_mapping['classes']
            else:
                class_names = list(self.PEST_DATABASE.keys())
            
            predicted_pest = class_names[predicted_class]
            
            # Apply thresholds
            success = (max_confidence >= self.confidence_threshold and 
                      uncertainty <= self.uncertainty_threshold)
            
            result = {
                'success': success,
                'pest_type': predicted_pest if success else 'unknown',
                'confidence': float(max_confidence),
                'uncertainty': float(uncertainty),
                'method': 'EfficientNet-B0 Ensemble',
                'models_used': len(self.efficientnet_models),
                'mc_samples': n_samples,
                'all_probabilities': {class_names[i]: float(mean_probs[i]) 
                                    for i in range(len(class_names))},
                'metadata': self._get_pest_metadata(predicted_pest) if success else None,
                'thresholds': {
                    'confidence': self.confidence_threshold,
                    'uncertainty': self.uncertainty_threshold
                }
            }
            
            logger.info(f"EfficientNet detection: {predicted_pest} "
                       f"(conf: {max_confidence:.3f}, unc: {uncertainty:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"EfficientNet detection failed: {e}")
            return self._create_error_result(f"EfficientNet error: {str(e)}")
    
    def _detect_with_basic(self, image):
        """Detect pest using basic CNN classifier."""
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Preprocess image
            input_tensor = self.basic_transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = self.basic_model(input_tensor)
                probs = F.softmax(output, dim=1)
                confidence, predicted = torch.max(probs, 1)
            
            max_confidence = float(confidence[0])
            predicted_class = int(predicted[0])
            
            class_names = list(self.PEST_DATABASE.keys())
            predicted_pest = class_names[predicted_class]
            
            success = max_confidence >= self.confidence_threshold
            
            result = {
                'success': success,
                'pest_type': predicted_pest if success else 'unknown',
                'confidence': max_confidence,
                'uncertainty': 1.0 - max_confidence,  # Simple uncertainty estimate
                'method': 'Basic CNN',
                'metadata': self._get_pest_metadata(predicted_pest) if success else None
            }
            
            logger.info(f"Basic CNN detection: {predicted_pest} (conf: {max_confidence:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"Basic CNN detection failed: {e}")
            return self._create_error_result(f"Basic CNN error: {str(e)}")
    
    def _get_pest_metadata(self, pest_type):
        """Get comprehensive metadata for detected pest."""
        if pest_type not in self.PEST_DATABASE:
            return None
        
        pest_info = self.PEST_DATABASE[pest_type]
        return {
            'scientific_name': pest_info['scientific_name'],
            'common_names': pest_info['common_names'],
            'damage_type': pest_info['damage_type'],
            'severity_indicators': pest_info['severity_indicators'],
            'affected_crops': pest_info['affected_crops'],
            'detection_features': pest_info['detection_features']
        }
    
    def _create_negative_result(self, reason):
        """Create a negative detection result."""
        return {
            'success': False,
            'pest_type': 'unknown',
            'confidence': 0.0,
            'uncertainty': 1.0,
            'reason': reason,
            'metadata': None
        }
    
    def _create_error_result(self, error_message):
        """Create an error result."""
        return {
            'success': False,
            'pest_type': 'error',
            'confidence': 0.0,
            'uncertainty': 1.0,
            'error': error_message,
            'metadata': None
        }
    
    def get_system_info(self):
        """Get information about the detection system capabilities."""
        return {
            'available_backends': self._get_available_backends(),
            'ml_capabilities': ML_CAPABILITIES,
            'efficientnet_models': len(self.efficientnet_models),
            'basic_model_available': self.basic_model is not None,
            'supported_pests': list(self.PEST_DATABASE.keys()),
            'thresholds': {
                'confidence': self.confidence_threshold,
                'uncertainty': self.uncertainty_threshold
            }
        }
    
    def update_thresholds(self, confidence_threshold=None, uncertainty_threshold=None):
        """Update detection thresholds."""
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
            logger.info(f"Updated confidence threshold to {confidence_threshold}")
        
        if uncertainty_threshold is not None:
            self.uncertainty_threshold = uncertainty_threshold
            logger.info(f"Updated uncertainty threshold to {uncertainty_threshold}")


# Backward compatibility aliases
PestDetector = UnifiedPestDetector
ImprovedPestDetector = UnifiedPestDetector
EnhancedPestDetector = UnifiedPestDetector


def main():
    """Test the unified pest detector."""
    detector = UnifiedPestDetector()
    
    # Print system information
    info = detector.get_system_info()
    print("Unified Pest Detection System")
    print("=" * 40)
    print(f"Available backends: {', '.join(info['available_backends'])}")
    print(f"ML capabilities: {info['ml_capabilities']}")
    print(f"Supported pests: {len(info['supported_pests'])}")
    print(f"Confidence threshold: {info['thresholds']['confidence']}")
    print(f"Uncertainty threshold: {info['thresholds']['uncertainty']}")
    
    # Test with a sample image if available
    test_images = [
        "datasets/ants/ants (1).jpg",
        "test_image.jpg"
    ]
    
    for test_image in test_images:
        if Path(test_image).exists():
            print(f"\nTesting with {test_image}:")
            result = detector.detect_pest(test_image)
            print(f"Result: {result}")
            break
    else:
        print("\nNo test images found. Please ensure test images are available.")


if __name__ == "__main__":
    main()
