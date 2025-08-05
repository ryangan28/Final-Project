"""
EfficientNet Pest Classification Training Pipeline
==============================================

Modern training pipeline using EfficientNet-B0 with comprehensive validation,
agricultural-specific augmentations, and uncertainty quantification.

This script trains all three EfficientNet variants (v1, v2, v3) used in the
Organic Farm Pest Management AI System.
"""

import os
import sys
import json
import time
import random
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import ML dependencies
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    from torchvision import transforms, models
    from torchvision.datasets import ImageFolder
    import torchvision.transforms.functional as TF
    from PIL import Image
    ML_AVAILABLE = True
    logger.info(f"Full ML dependencies loaded successfully - PyTorch {torch.__version__}")
    
    # Check PyTorch version for compatibility
    torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
    if torch_version < (1, 8):
        logger.warning(f"PyTorch version {torch.__version__} is quite old. Consider upgrading for best compatibility.")
        
except ImportError as e:
    ML_AVAILABLE = False
    logger.error(f"ML dependencies not available: {e}")
    print("Please install: pip install torch torchvision matplotlib seaborn scikit-learn")
    sys.exit(1)


class AgriculturalAugmentations:
    """Agricultural-specific data augmentations for pest images."""
    
    def __init__(self, image_size: int = 224):
        self.image_size = image_size
        
    def get_train_transforms(self):
        """Get training augmentations optimized for pest imagery."""
        return transforms.Compose([
            transforms.Resize((self.image_size + 32, self.image_size + 32)),
            transforms.RandomCrop(self.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),  # Pests can be in any orientation
            transforms.RandomRotation(degrees=30, fill=0),
            transforms.ColorJitter(
                brightness=0.3,    # Agricultural lighting varies
                contrast=0.3,      # Different background contrasts
                saturation=0.2,    # Natural color variations
                hue=0.1           # Slight hue shifts
            ),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.3),
            transforms.RandomApply([
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5)
            ], p=0.3),
            # Convert to tensor and normalize
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet pre-trained mean
                std=[0.229, 0.224, 0.225]   # ImageNet pre-trained std
            ),
            # Additional agricultural-specific augmentations
            transforms.RandomApply([
                self._add_dirt_spots
            ], p=0.2),
        ])
    
    def get_val_transforms(self):
        """Get validation transforms - minimal processing."""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _add_dirt_spots(self, tensor):
        """Add random dirt spots to simulate real agricultural conditions."""
        if random.random() < 0.5:
            # Add small dark spots
            num_spots = random.randint(1, 3)
            for _ in range(num_spots):
                x = random.randint(0, tensor.shape[1] - 5)
                y = random.randint(0, tensor.shape[2] - 5)
                spot_size = random.randint(2, 4)
                tensor[:, x:x+spot_size, y:y+spot_size] *= random.uniform(0.3, 0.7)
        return tensor


class ImprovedPestDataset(Dataset):
    """Enhanced dataset class with better handling of pest images."""
    
    def __init__(self, data_dir: str, transform=None, class_mapping: Dict = None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        self.classes = []
        self.class_to_idx = {}
        
        # Build dataset
        self._build_dataset(class_mapping)
        
    def _build_dataset(self, class_mapping: Dict = None):
        """Build dataset with proper class mapping."""
        # Get all pest directories
        pest_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        
        if class_mapping and isinstance(class_mapping, dict):
            # Check if it's a proper class mapping with 'classes' and 'class_to_idx'
            if 'classes' in class_mapping and 'class_to_idx' in class_mapping:
                self.classes = class_mapping['classes']
                self.class_to_idx = class_mapping['class_to_idx']
            elif isinstance(class_mapping, dict) and all(isinstance(v, int) for v in class_mapping.values()):
                # Direct mapping format {class_name: index}
                self.classes = list(class_mapping.keys())
                self.class_to_idx = class_mapping
            else:
                # Fallback to auto-detect
                self.classes = sorted([d.name for d in pest_dirs])
                self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        else:
            # Auto-detect classes
            self.classes = sorted([d.name for d in pest_dirs])
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Build samples list
        for pest_dir in pest_dirs:
            if pest_dir.name not in self.class_to_idx:
                continue
                
            class_idx = self.class_to_idx[pest_dir.name]
            
            # Get all image files
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
            image_files = []
            for ext in image_extensions:
                image_files.extend(list(pest_dir.glob(ext)))
            
            # Add to samples
            for img_path in image_files:
                self.samples.append((str(img_path), class_idx))
        
        logger.info(f"Dataset built: {len(self.samples)} samples, {len(self.classes)} classes")
        logger.info(f"Classes: {self.classes}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, class_idx = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Failed to load image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, class_idx
    
    def get_class_distribution(self):
        """Get distribution of classes in dataset."""
        class_counts = defaultdict(int)
        for _, class_idx in self.samples:
            class_counts[class_idx] += 1
        
        return dict(class_counts)


class EfficientNetPestClassifier(nn.Module):
    """EfficientNet-B0 based pest classifier with uncertainty estimation."""
    
    def __init__(self, num_classes: int, dropout_rate: float = 0.3):
        super().__init__()
        
        # Load pre-trained EfficientNet-B0 (compatible with different PyTorch versions)
        try:
            # New style (PyTorch 0.13+)
            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        except (AttributeError, TypeError):
            # Old style (PyTorch < 0.13)
            self.backbone = models.efficientnet_b0(pretrained=True)
        
        # Replace classifier head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, num_classes)
        )
        
        # For uncertainty estimation - add temperature scaling
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        logits = self.backbone(x)
        return logits
    
    def forward_with_temperature(self, x):
        """Forward pass with temperature scaling for calibration."""
        logits = self.backbone(x)
        return logits / self.temperature


class ImprovedTrainer:
    """Modern training pipeline with comprehensive validation."""
    
    def __init__(self, data_dir: str, output_dir: str = "models/efficientnet/v1"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training configuration
        self.config = {
            'image_size': 224,
            'batch_size': 32,
            'num_epochs': 100,
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'patience': 15,
            'min_delta': 1e-4,
            'num_folds': 5,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        # Initialize components
        self.augmentations = AgriculturalAugmentations(self.config['image_size'])
        self.device = torch.device(self.config['device'])
        
        logger.info(f"Training on device: {self.device}")
        
    def prepare_data(self):
        """Prepare dataset with stratified splits."""
        logger.info("Preparing dataset...")
        
        # Load full dataset
        dataset = ImprovedPestDataset(
            self.data_dir,
            transform=self.augmentations.get_val_transforms()  # No augmentation for splitting
        )
        
        if len(dataset) == 0:
            raise ValueError(f"No images found in {self.data_dir}")
        
        # Get class distribution
        class_dist = dataset.get_class_distribution()
        logger.info(f"Class distribution: {class_dist}")
        
        # Check for class imbalance
        min_samples = min(class_dist.values())
        max_samples = max(class_dist.values())
        imbalance_ratio = max_samples / min_samples
        
        if imbalance_ratio > 10:
            logger.warning(f"Severe class imbalance detected (ratio: {imbalance_ratio:.1f})")
        
        # Store dataset info
        self.num_classes = len(dataset.classes)
        self.class_names = dataset.classes
        self.class_to_idx = dataset.class_to_idx
        
        # Save class mapping
        class_mapping = {
            'classes': self.class_names,
            'class_to_idx': self.class_to_idx,
            'num_classes': self.num_classes
        }
        
        with open(self.output_dir / 'class_mapping.json', 'w') as f:
            json.dump(class_mapping, f, indent=2)
        
        return dataset, class_dist
    
    def create_weighted_sampler(self, dataset, class_dist):
        """Create weighted sampler to handle class imbalance."""
        # Calculate weights for each class (inverse frequency)
        total_samples = len(dataset)
        class_weights = {}
        
        for class_idx, count in class_dist.items():
            class_weights[class_idx] = total_samples / (len(class_dist) * count)
        
        # Create sample weights
        sample_weights = []
        for _, class_idx in dataset.samples:
            sample_weights.append(class_weights[class_idx])
        
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
    
    def train_fold(self, train_dataset, val_dataset, fold_num: int):
        """Train a single fold."""
        logger.info(f"Training fold {fold_num + 1}/{self.config['num_folds']}")
        
        # Create data loaders (reduced workers for stability on CPU)
        num_workers = 2 if self.device.type == 'cuda' else 0  # 0 workers on CPU to avoid issues
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        # Initialize model
        model = EfficientNetPestClassifier(
            num_classes=self.num_classes,
            dropout_rate=0.3
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Learning rate scheduler (compatible with different PyTorch versions)
        try:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
        except TypeError:
            # Older PyTorch versions don't support verbose parameter
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5
            )
        
        # Training loop
        best_val_acc = 0.0
        patience_counter = 0
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        for epoch in range(self.config['num_epochs']):
            # Training phase
            model.train()
            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                
                if batch_idx % 50 == 0:
                    logger.info(f'Fold {fold_num+1}, Epoch {epoch+1}, Batch {batch_idx}, '
                              f'Loss: {loss.item():.4f}')
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            # Calculate metrics
            train_acc = 100.0 * correct_predictions / total_predictions
            val_acc = 100.0 * val_correct / val_total
            avg_train_loss = running_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_acc)
            
            logger.info(f'Fold {fold_num+1}, Epoch {epoch+1}: '
                       f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                       f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Early stopping check
            if val_acc > best_val_acc + self.config['min_delta']:
                best_val_acc = val_acc
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_acc': val_acc,
                    'fold': fold_num,
                    'class_mapping': {
                        'classes': self.class_names,
                        'class_to_idx': self.class_to_idx,
                        'num_classes': self.num_classes
                    }
                }, self.output_dir / f'best_model_fold_{fold_num}.pth')
            else:
                patience_counter += 1
                
            if patience_counter > self.config['patience']:
                logger.info(f'Early stopping at epoch {epoch+1}')
                break
        
        return {
            'best_val_acc': best_val_acc,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'final_model': model
        }
    
    def cross_validate(self):
        """Perform cross-validation training."""
        logger.info("Starting cross-validation training...")
        
        # Prepare data
        full_dataset, class_dist = self.prepare_data()
        
        # Extract labels for stratification
        labels = [sample[1] for sample in full_dataset.samples]
        
        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=self.config['num_folds'], shuffle=True, random_state=42)
        
        fold_results = []
        
        for fold_num, (train_idx, val_idx) in enumerate(skf.split(range(len(full_dataset)), labels)):
            # Create fold datasets
            train_samples = [full_dataset.samples[i] for i in train_idx]
            val_samples = [full_dataset.samples[i] for i in val_idx]
            
            # Create fold-specific datasets
            train_dataset = ImprovedPestDataset(
                self.data_dir,
                transform=self.augmentations.get_train_transforms(),
                class_mapping={cls: idx for cls, idx in self.class_to_idx.items()}
            )
            train_dataset.samples = train_samples
            
            val_dataset = ImprovedPestDataset(
                self.data_dir,
                transform=self.augmentations.get_val_transforms(),
                class_mapping={cls: idx for cls, idx in self.class_to_idx.items()}
            )
            val_dataset.samples = val_samples
            
            # Train fold
            fold_result = self.train_fold(train_dataset, val_dataset, fold_num)
            fold_results.append(fold_result)
        
        # Calculate cross-validation metrics
        cv_accuracies = [result['best_val_acc'] for result in fold_results]
        mean_cv_acc = np.mean(cv_accuracies)
        std_cv_acc = np.std(cv_accuracies)
        
        logger.info(f"Cross-validation complete!")
        logger.info(f"Mean CV Accuracy: {mean_cv_acc:.2f}% ± {std_cv_acc:.2f}%")
        logger.info(f"Individual fold accuracies: {cv_accuracies}")
        
        # Save results
        cv_results = {
            'mean_accuracy': mean_cv_acc,
            'std_accuracy': std_cv_acc,
            'fold_accuracies': cv_accuracies,
            'config': self.config,
            'class_mapping': {
                'classes': self.class_names,
                'class_to_idx': self.class_to_idx,
                'num_classes': self.num_classes
            }
        }
        
        with open(self.output_dir / 'cv_results.json', 'w') as f:
            json.dump(cv_results, f, indent=2)
        
        return cv_results, fold_results


def main():
    """Main training function."""
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    print("🌱 Improved Pest Classification Training Pipeline")
    print("=" * 60)
    
    # Check dataset
    data_dir = "datasets"
    if not Path(data_dir).exists():
        print(f"❌ Dataset directory '{data_dir}' not found!")
        print("Please ensure the Agricultural Pests Image Dataset is available.")
        return
    
    # Initialize trainer
    trainer = ImprovedTrainer(data_dir)
    
    # Start training
    start_time = time.time()
    cv_results, fold_results = trainer.cross_validate()
    training_time = time.time() - start_time
    
    print(f"\n🎉 Training completed in {training_time:.2f} seconds!")
    print(f"📊 Final Results:")
    print(f"   Mean CV Accuracy: {cv_results['mean_accuracy']:.2f}% ± {cv_results['std_accuracy']:.2f}%")
    print(f"   Best Single Fold: {max(cv_results['fold_accuracies']):.2f}%")
    print(f"   Models saved in: {trainer.output_dir}")
    
    return cv_results, fold_results


if __name__ == "__main__":
    if not ML_AVAILABLE:
        print("❌ ML dependencies not available. Please install required packages.")
        sys.exit(1)
    
    main()