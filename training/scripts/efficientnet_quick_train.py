"""
EfficientNet Quick Training Script
================================

Fast training script for testing the EfficientNet pipeline with reduced parameters.
Perfect for development and validation of the Organic Farm Pest Management AI System.
"""

import os
import sys
import time
import random
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def quick_improved_train():
    """Quick training function for development and testing."""
    
    print("🚀 Quick Improved Pest Classification Training")
    print("=" * 60)
    print("📋 Using EfficientNet-B0 with agricultural augmentations")
    print("⚡ Reduced parameters for fast development testing")
    print()
    
    # Check if datasets directory exists
    datasets_dir = Path("datasets")
    if not datasets_dir.exists():
        print("❌ datasets/ directory not found!")
        print("   Please ensure the Agricultural Pests Image Dataset is in datasets/")
        print("   You can download it from: https://www.kaggle.com/datasets/vencerlanz09/agricultural-pests-image-dataset")
        return False
    
    # Count images and validate dataset
    total_images = 0
    pest_classes = []
    
    print("📊 Scanning dataset...")
    for pest_dir in datasets_dir.iterdir():
        if pest_dir.is_dir():
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
            image_count = 0
            for ext in image_extensions:
                image_count += len(list(pest_dir.glob(ext)))
            
            if image_count > 0:
                pest_classes.append(pest_dir.name)
                total_images += image_count
                print(f"   ✅ {pest_dir.name}: {image_count} images")
            else:
                print(f"   ⚠️ {pest_dir.name}: No images found")
    
    print(f"\n📈 Dataset Summary:")
    print(f"   Total Images: {total_images}")
    print(f"   Pest Classes: {len(pest_classes)}")
    print(f"   Classes: {', '.join(pest_classes)}")
    
    if total_images < 50:
        print("❌ Not enough images for training (minimum 50 recommended for quick training)")
        print("   For production training, use at least 100 images per class")
        return False
    
    if len(pest_classes) < 2:
        print("❌ Need at least 2 pest classes for classification")
        return False
    
    # Check ML dependencies
    try:
        import torch
        import torchvision
        print(f"✅ PyTorch available: {torch.__version__}")
        print(f"✅ Torchvision available: {torchvision.__version__}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ Training device: {device}")
        
        if device.type == 'cuda':
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
    except ImportError as e:
        print(f"❌ ML dependencies not available: {e}")
        print("Please install: pip install torch torchvision matplotlib seaborn scikit-learn")
        return False
    
    # Import training components
    try:
        from training.scripts.efficientnet_train import ImprovedTrainer
        print("✅ Improved training components loaded")
    except ImportError as e:
        print(f"❌ Failed to import improved training components: {e}")
        return False
    
    # Configure quick training parameters
    print("\n⚙️ Quick Training Configuration:")
    print("   • Model: EfficientNet-B0")
    print("   • Cross-validation: 3 folds (reduced)")
    print("   • Epochs per fold: 20 (reduced)")  
    print("   • Patience: 8 (reduced)")
    print("   • Batch size: 16 (reduced)")
    print("   • Agricultural augmentations: Enabled")
    print("   • Uncertainty quantification: Enabled")
    
    # Initialize trainer with quick settings
    trainer = ImprovedTrainer(data_dir=str(datasets_dir), output_dir="models/archive/improved_quick")
    
    # Override config for quick training
    trainer.config.update({
        'num_epochs': 20,        # Reduced from 100
        'patience': 8,           # Reduced from 15
        'num_folds': 3,          # Reduced from 5
        'batch_size': 16,        # Reduced from 32
        'learning_rate': 2e-4    # Slightly higher for faster convergence
    })
    
    print(f"\n🏗️ Output directory: {trainer.output_dir}")
    
    # Confirm training start
    print("\n" + "="*60)
    estimated_time = "10-30 minutes"
    if device.type == 'cuda':
        estimated_time = "5-15 minutes"
    
    print(f"⏱️ Estimated training time: {estimated_time}")
    print(f"💾 Models will be saved to: {trainer.output_dir}")
    print("="*60)
    
    # Start training
    start_time = time.time()
    
    try:
        print("\n🚀 Starting quick improved training...")
        cv_results, fold_results = trainer.cross_validate()
        
        training_time = time.time() - start_time
        
        print(f"\n🎉 Quick training completed in {training_time:.1f} seconds ({training_time/60:.1f} minutes)!")
        
        # Display results
        print(f"\n📊 Quick Training Results:")
        print(f"   Mean CV Accuracy: {cv_results['mean_accuracy']:.2f}% ± {cv_results['std_accuracy']:.2f}%")
        print(f"   Best Single Fold: {max(cv_results['fold_accuracies']):.2f}%")
        print(f"   Worst Single Fold: {min(cv_results['fold_accuracies']):.2f}%")
        print(f"   Individual Accuracies: {[f'{acc:.1f}%' for acc in cv_results['fold_accuracies']]}")
        
        # Production readiness assessment
        print(f"\n🎯 Quick Assessment:")
        mean_acc = cv_results['mean_accuracy']
        std_acc = cv_results['std_accuracy']
        
        if mean_acc >= 80:
            print("   ✅ Good accuracy for agricultural pest detection")
        elif mean_acc >= 70:
            print("   ⚠️ Moderate accuracy - consider more training data or longer training")
        else:
            print("   ❌ Low accuracy - needs more data or hyperparameter tuning")
        
        if std_acc <= 5:
            print("   ✅ Good stability across folds")
        else:
            print("   ⚠️ High variance - consider more consistent data or regularization")
        
        # Next steps
        print(f"\n🎯 Next Steps:")
        print("   1. Test the models: python training/scripts/evaluate_models.py")
        print("   2. Use in main system: The EfficientNet detector will automatically load these models")
        print("   3. For production: Run full training with more epochs and folds")
        
        print(f"\n📁 Models saved in: {trainer.output_dir}")
        
        return True
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        logger.exception("Training failed with exception:")
        return False


def main():
    """Main entry point."""
    # Set random seeds for reproducibility
    random.seed(42)
    try:
        import numpy as np
        np.random.seed(42)
        
        import torch
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            torch.cuda.manual_seed_all(42)
    except ImportError:
        pass
    
    success = quick_improved_train()
    
    if success:
        print("\n🎊 Quick improved training completed successfully!")
        print("🔬 Ready to test improved models with uncertainty quantification")
    else:
        print("\n💥 Quick training failed. Please check the error messages above.")
    
    return success


if __name__ == "__main__":
    main()