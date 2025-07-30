"""
Quick Training Script for YOLOv8 Pest Classification
===================================================

Run this script to train a YOLOv8-nano model on the pest dataset.
"""

import os
import sys
from pathlib import Path

def quick_train():
    """Quick training function that can be run immediately."""
    
    print("🌱 YOLOv8 Pest Classification Training")
    print("=" * 50)
    
    # Check if datasets directory exists
    if not Path("datasets").exists():
        print("❌ datasets/ directory not found!")
        print("   Please ensure the Agricultural Pests Image Dataset is in datasets/")
        return False
    
    # Count images
    total_images = 0
    pest_classes = []
    
    for pest_dir in Path("datasets").iterdir():
        if pest_dir.is_dir():
            image_count = len(list(pest_dir.glob("*.jpg")) + list(pest_dir.glob("*.png")) + list(pest_dir.glob("*.jpeg")))
            if image_count > 0:
                pest_classes.append(pest_dir.name)
                total_images += image_count
                print(f"   {pest_dir.name}: {image_count} images")
    
    print(f"\n📊 Found {total_images} images across {len(pest_classes)} pest classes")
    
    if total_images < 100:
        print("❌ Not enough images for training (minimum 100 recommended)")
        return False
    
    # Install ultralytics if not available
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics YOLOv8 is available")
    except ImportError:
        print("📦 Installing ultralytics...")
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
            print("✅ Ultralytics installed successfully")
            from ultralytics import YOLO
        except Exception as e:
            print(f"❌ Failed to install ultralytics: {e}")
            print("Please install manually: pip install ultralytics")
            return False
    
    # Prepare dataset in YOLO format
    print("\n🏗️ Preparing dataset for YOLO training...")
    
    try:
        # Create a simple classification dataset
        # YOLOv8 can work directly with image folders for classification
        
        # Load YOLOv8n-cls model
        print("📥 Loading YOLOv8-nano classification model...")
        model = YOLO('yolov8n-cls.pt')  # Automatically downloads if not present
        
        # Train the model
        print("🚀 Starting training...")
        
        # Check device availability
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        batch_size = 16 if device == 'cuda' else 8  # Smaller batch for CPU
        
        print(f"   - Device: {device}")
        print("   - Epochs: 30 (quick training)")
        print("   - Image size: 224x224")
        print(f"   - Batch size: {batch_size}")
        
        results = model.train(
            data='datasets',  # Point directly to datasets folder
            epochs=30,  # Quick training
            imgsz=224,
            batch=batch_size,
            device=device,
            project='models',
            name='pest_classifier',
            save=True,
            plots=True,
            verbose=True,
            patience=10,  # Early stopping
            save_period=5  # Save every 5 epochs
        )
        
        # Get the best model path
        best_model = results.save_dir / 'weights' / 'best.pt'
        
        print(f"\n🎉 Training complete!")
        print(f"   Best model saved: {best_model}")
        
        # Validate the model
        print("\n📊 Running validation...")
        val_results = model.val()
        
        print(f"   Top-1 Accuracy: {val_results.top1:.3f}")
        print(f"   Top-5 Accuracy: {val_results.top5:.3f}")
        
        # Export to ONNX for deployment
        print("\n📤 Exporting to ONNX...")
        try:
            onnx_path = model.export(format='onnx', optimize=True, simplify=True)
            print(f"   ONNX model: {onnx_path}")
        except Exception as e:
            print(f"   ONNX export failed: {e}")
        
        # Copy best model to main directory for easy access
        import shutil
        main_model_path = Path("pest_model_yolov8n.pt")
        shutil.copy2(best_model, main_model_path)
        print(f"\n✅ Model copied to: {main_model_path}")
        
        print("\n🔧 To use the trained model:")
        print("   1. Update vision/pest_detector.py to use the trained model")
        print("   2. Set model_path='pest_model_yolov8n.pt' in PestDetector initialization")
        print("   3. The system will automatically use the trained model instead of simulation")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("\nTroubleshooting:")
        print("   - Ensure you have enough disk space")
        print("   - Check if you have GPU drivers installed (for faster training)")
        print("   - Try reducing batch size if out of memory")
        return False

def test_trained_model():
    """Test the trained model with a sample image."""
    model_path = Path("pest_model_yolov8n.pt")
    
    if not model_path.exists():
        print("❌ Trained model not found. Run training first.")
        return
    
    try:
        from ultralytics import YOLO
        
        print("\n🧪 Testing trained model...")
        model = YOLO(str(model_path))
        
        # Find a test image
        test_image = None
        for pest_dir in Path("datasets").iterdir():
            if pest_dir.is_dir():
                images = list(pest_dir.glob("*.jpg")) + list(pest_dir.glob("*.png"))
                if images:
                    test_image = images[0]
                    expected_class = pest_dir.name
                    break
        
        if test_image:
            print(f"   Testing with: {test_image}")
            print(f"   Expected class: {expected_class}")
            
            results = model(str(test_image))
            
            if results:
                result = results[0]
                if hasattr(result, 'probs'):
                    top_class = result.names[result.probs.top1]
                    confidence = float(result.probs.top1conf)
                    
                    print(f"   Predicted: {top_class}")
                    print(f"   Confidence: {confidence:.3f}")
                    
                    if top_class.lower() == expected_class.lower():
                        print("   ✅ Correct prediction!")
                    else:
                        print("   ⚠️ Different prediction (may still be valid)")
        else:
            print("   ❌ No test images found")
            
    except Exception as e:
        print(f"   ❌ Test failed: {e}")

if __name__ == "__main__":
    print("YOLOv8 Pest Classification Training Script")
    print("==========================================")
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_trained_model()
    else:
        success = quick_train()
        if success:
            print("\n" + "="*50)
            print("🎉 SUCCESS! Your pest classification model is ready!")
            print("="*50)
            
            # Test the model
            test_trained_model()
        else:
            print("\n❌ Training failed. Please check the errors above.")
