"""
YOLOv8-nano Training Pipeline for Pest Classification
===================================================

This module implements a complete training pipeline using YOLOv8-nano
for agricultural pest classification using the datasets.
"""

import os
import yaml
import shutil
from pathlib import Path
import random
from typing import Dict, List, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PestDatasetPreparer:
    """Prepare the pest dataset for YOLOv8 training."""
    
    def __init__(self, source_dir: str = "datasets", output_dir: str = "yolo_dataset"):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.classes = []
        
    def prepare_dataset(self, train_split: float = 0.8, val_split: float = 0.15) -> Dict:
        """
        Prepare dataset in YOLOv8 format.
        
        Args:
            train_split: Fraction for training set
            val_split: Fraction for validation set (remaining goes to test)
            
        Returns:
            Dictionary with dataset statistics
        """
        logger.info("🔄 Preparing pest dataset for YOLOv8 training...")
        
        # Create output directories
        splits = ['train', 'val', 'test']
        for split in splits:
            (self.output_dir / split).mkdir(parents=True, exist_ok=True)
        
        # Get class names from directory structure
        self.classes = sorted([d.name for d in self.source_dir.iterdir() if d.is_dir()])
        logger.info(f"📁 Found {len(self.classes)} pest classes: {self.classes}")
        
        stats = {
            'total_images': 0,
            'train_images': 0,
            'val_images': 0,
            'test_images': 0,
            'classes': self.classes
        }
        
        # Process each class
        for class_idx, class_name in enumerate(self.classes):
            class_dir = self.source_dir / class_name
            if not class_dir.exists():
                continue
                
            # Get all images in class
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                image_files.extend(list(class_dir.glob(ext)))
            
            if not image_files:
                logger.warning(f"⚠️ No images found in {class_name}")
                continue
            
            # Shuffle images
            random.shuffle(image_files)
            total_images = len(image_files)
            
            # Calculate split indices
            train_end = int(total_images * train_split)
            val_end = int(total_images * (train_split + val_split))
            
            # Split images
            train_images = image_files[:train_end]
            val_images = image_files[train_end:val_end]
            test_images = image_files[val_end:]
            
            # Copy images to respective directories
            for split, images in [('train', train_images), ('val', val_images), ('test', test_images)]:
                split_dir = self.output_dir / split
                for img_path in images:
                    # Copy image with class prefix
                    new_name = f"{class_name}_{img_path.name}"
                    shutil.copy2(img_path, split_dir / new_name)
            
            # Update statistics
            stats['total_images'] += total_images
            stats['train_images'] += len(train_images)
            stats['val_images'] += len(val_images)
            stats['test_images'] += len(test_images)
            
            logger.info(f"✅ {class_name}: {total_images} images "
                       f"(train: {len(train_images)}, val: {len(val_images)}, test: {len(test_images)})")
        
        # Create data.yaml for YOLOv8
        self._create_data_yaml()
        
        logger.info("🎉 Dataset preparation complete!")
        return stats
    
    def _create_data_yaml(self):
        """Create data.yaml configuration file for YOLOv8."""
        data_config = {
            'path': str(self.output_dir.absolute()),
            'train': 'train',
            'val': 'val',
            'test': 'test',
            'nc': len(self.classes),  # number of classes
            'names': self.classes
        }
        
        yaml_path = self.output_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
        
        logger.info(f"📝 Created {yaml_path}")


class YOLOv8PestTrainer:
    """Train YOLOv8-nano model for pest classification."""
    
    def __init__(self, dataset_dir: str = "yolo_dataset"):
        self.dataset_dir = Path(dataset_dir)
        self.model = None
        
    def install_requirements(self):
        """Install required packages for YOLOv8 training."""
        import subprocess
        import sys
        
        requirements = [
            'ultralytics>=8.0.0',
            'torch>=1.8.0',
            'torchvision>=0.9.0',
            'opencv-python>=4.5.0',
            'pillow>=8.0.0',
            'matplotlib>=3.3.0',
            'seaborn>=0.11.0'
        ]
        
        logger.info("📦 Installing YOLOv8 requirements...")
        for package in requirements:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                logger.info(f"✅ Installed {package}")
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Failed to install {package}: {e}")
                return False
        
        return True
    
    def train_model(self, epochs: int = 100, batch_size: int = 16, image_size: int = 224) -> str:
        """
        Train YOLOv8-nano classification model.
        
        Args:
            epochs: Number of training epochs
            batch_size: Training batch size
            image_size: Input image size
            
        Returns:
            Path to trained model
        """
        try:
            from ultralytics import YOLO
            logger.info("🚀 Starting YOLOv8-nano training...")
            
            # Load YOLOv8-nano classification model
            self.model = YOLO('models/yolov8n-cls.pt')  # nano classification model
            
            # Training configuration
            data_yaml = self.dataset_dir / 'data.yaml'
            
            # Train the model
            results = self.model.train(
                data=str(data_yaml),
                epochs=epochs,
                batch=batch_size,
                imgsz=image_size,
                device='auto',  # Use GPU if available, otherwise CPU
                project='pest_training',
                name='yolov8n_pest_classifier',
                save=True,
                plots=True,
                verbose=True
            )
            
            # Get best model path
            best_model_path = results.save_dir / 'weights' / 'best.pt'
            
            logger.info(f"🎉 Training complete! Best model saved at: {best_model_path}")
            return str(best_model_path)
            
        except ImportError:
            logger.error("❌ Ultralytics not installed. Run install_requirements() first.")
            return None
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            return None
    
    def validate_model(self, model_path: str) -> Dict:
        """Validate the trained model."""
        try:
            from ultralytics import YOLO
            
            model = YOLO(model_path)
            
            # Run validation
            results = model.val(
                data=str(self.dataset_dir / 'data.yaml'),
                plots=True,
                save_json=True
            )
            
            # Extract metrics
            metrics = {
                'top1_accuracy': results.top1,
                'top5_accuracy': results.top5,
                'total_params': sum(p.numel() for p in model.model.parameters()),
                'model_size_mb': Path(model_path).stat().st_size / (1024 * 1024)
            }
            
            logger.info(f"📊 Validation Results:")
            logger.info(f"   Top-1 Accuracy: {metrics['top1_accuracy']:.4f}")
            logger.info(f"   Top-5 Accuracy: {metrics['top5_accuracy']:.4f}")
            logger.info(f"   Model Size: {metrics['model_size_mb']:.2f} MB")
            logger.info(f"   Parameters: {metrics['total_params']:,}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            return {}
    
    def export_model(self, model_path: str, formats: List[str] = ['onnx', 'tflite']) -> Dict[str, str]:
        """
        Export trained model to different formats for deployment.
        
        Args:
            model_path: Path to trained model
            formats: Export formats ['onnx', 'tflite', 'openvino', 'coreml']
            
        Returns:
            Dictionary mapping format to exported model path
        """
        try:
            from ultralytics import YOLO
            
            model = YOLO(model_path)
            exported_models = {}
            
            for fmt in formats:
                logger.info(f"📤 Exporting to {fmt.upper()}...")
                
                export_path = model.export(
                    format=fmt,
                    imgsz=224,
                    optimize=True,
                    simplify=True if fmt == 'onnx' else False
                )
                
                exported_models[fmt] = str(export_path)
                logger.info(f"✅ {fmt.upper()} model exported: {export_path}")
            
            return exported_models
            
        except Exception as e:
            logger.error(f"❌ Export failed: {e}")
            return {}


def main():
    """Complete training pipeline."""
    # Set random seed for reproducibility
    random.seed(42)
    
    # Step 1: Prepare dataset
    logger.info("🏗️ Step 1: Preparing dataset...")
    preparer = PestDatasetPreparer()
    stats = preparer.prepare_dataset()
    
    print("\n📊 Dataset Statistics:")
    print(f"   Total Images: {stats['total_images']}")
    print(f"   Training: {stats['train_images']}")
    print(f"   Validation: {stats['val_images']}")
    print(f"   Test: {stats['test_images']}")
    print(f"   Classes: {len(stats['classes'])}")
    
    # Step 2: Install requirements
    logger.info("\n🏗️ Step 2: Installing requirements...")
    trainer = YOLOv8PestTrainer()
    if not trainer.install_requirements():
        logger.error("❌ Failed to install requirements")
        return
    
    # Step 3: Train model
    logger.info("\n🏗️ Step 3: Training YOLOv8-nano model...")
    model_path = trainer.train_model(epochs=50, batch_size=16)  # Reduced epochs for faster training
    
    if not model_path:
        logger.error("❌ Training failed")
        return
    
    # Step 4: Validate model
    logger.info("\n🏗️ Step 4: Validating model...")
    metrics = trainer.validate_model(model_path)
    
    # Step 5: Export model
    logger.info("\n🏗️ Step 5: Exporting model...")
    exported_models = trainer.export_model(model_path, ['onnx'])
    
    print("\n🎉 Training Pipeline Complete!")
    print(f"   Best Model: {model_path}")
    print(f"   Exported Models: {exported_models}")
    
    return model_path, metrics, exported_models


if __name__ == "__main__":
    main()
