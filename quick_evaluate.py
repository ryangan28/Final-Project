"""
Quick Model Evaluation Script
============================

Fast evaluation using a small subset of data for quick feedback.
"""

import sys
import time
import logging
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quick_evaluate():
    """Quick evaluation of trained models."""
    
    print("⚡ Quick Model Evaluation")
    print("=" * 40)
    
    # Check for ML dependencies
    try:
        import torch
        import torch.nn.functional as F
        from training.improved_train import ImprovedPestDataset, AgriculturalAugmentations, EfficientNetPestClassifier
        from torch.utils.data import DataLoader
        print("✅ ML dependencies loaded")
    except ImportError as e:
        print(f"❌ ML dependencies not available: {e}")
        return
    
    # Find trained models
    model_dirs = [Path("models/improved_quick"), Path("models/improved")]
    model_files = []
    model_dir = None
    
    for directory in model_dirs:
        if directory.exists():
            files = list(directory.glob('best_model_fold_*.pth'))
            if files:
                model_dir = directory
                model_files = files
                break
    
    if not model_files:
        print("❌ No trained models found")
        return
    
    print(f"📁 Found {len(model_files)} models in {model_dir}")
    
    # Load dataset (small subset)
    dataset_dir = Path("datasets")
    if not dataset_dir.exists():
        print("❌ Dataset not found")
        return
    
    augmentations = AgriculturalAugmentations()
    dataset = ImprovedPestDataset(
        str(dataset_dir),
        transform=augmentations.get_val_transforms()
    )
    
    if len(dataset) == 0:
        print("❌ Dataset is empty")
        return
    
    # Use small subset for quick evaluation (50 samples)
    total_samples = len(dataset.samples)
    subset_size = min(50, total_samples)
    np.random.seed(42)
    subset_indices = np.random.choice(total_samples, size=subset_size, replace=False)
    subset_samples = [dataset.samples[i] for i in subset_indices]
    dataset.samples = subset_samples
    
    print(f"📊 Using {len(dataset)} samples for quick evaluation")
    
    # Create data loader
    test_loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")
    
    # Evaluate each model
    results = {}
    
    for i, model_path in enumerate(model_files):
        print(f"\n🧪 Evaluating model {i+1}/{len(model_files)}: {model_path.name}")
        
        try:
            # Load model
            checkpoint = torch.load(model_path, map_location=device)
            num_classes = checkpoint['class_mapping']['num_classes']
            
            model = EfficientNetPestClassifier(num_classes=num_classes)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            
            # Evaluation
            correct = 0
            total = 0
            all_predictions = []
            all_labels = []
            inference_times = []
            
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    
                    # Measure inference time
                    start_time = time.time()
                    outputs = model(images)
                    inference_time = (time.time() - start_time) * 1000  # ms
                    inference_times.append(inference_time)
                    
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            # Calculate metrics
            accuracy = 100.0 * correct / total
            avg_inference_time = np.mean(inference_times)
            
            results[f'fold_{i}'] = {
                'accuracy': accuracy,
                'inference_time_ms': avg_inference_time,
                'model_path': str(model_path),
                'val_acc_training': checkpoint.get('val_acc', 0.0)
            }
            
            print(f"   ✅ Accuracy: {accuracy:.1f}%")
            print(f"   ⚡ Inference: {avg_inference_time:.1f}ms per batch")
            print(f"   📈 Training val acc: {checkpoint.get('val_acc', 0.0):.1f}%")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results[f'fold_{i}'] = {'error': str(e)}
    
    # Summary
    print(f"\n" + "="*40)
    print("📋 QUICK EVALUATION SUMMARY")
    print("="*40)
    
    successful_results = [r for r in results.values() if 'accuracy' in r]
    
    if successful_results:
        accuracies = [r['accuracy'] for r in successful_results]
        inference_times = [r['inference_time_ms'] for r in successful_results]
        
        print(f"✅ {len(successful_results)}/{len(results)} models evaluated successfully")
        print(f"📊 Accuracy: {np.mean(accuracies):.1f}% ± {np.std(accuracies):.1f}%")
        print(f"   Best: {max(accuracies):.1f}%, Worst: {min(accuracies):.1f}%")
        print(f"⚡ Inference: {np.mean(inference_times):.1f}ms avg per batch")
        print(f"🎯 Throughput: ~{8000/np.mean(inference_times):.0f} images/second")
        
        # Assessment
        mean_acc = np.mean(accuracies)
        if mean_acc >= 90:
            print("🏆 Excellent performance - production ready!")
        elif mean_acc >= 80:
            print("✅ Good performance - ready for deployment")
        elif mean_acc >= 70:
            print("⚠️ Moderate performance - consider improvements")
        else:
            print("❌ Low performance - needs significant improvements")
        
        print(f"\n💡 Note: This is a quick evaluation on {subset_size} samples.")
        print("   For comprehensive evaluation, run: python training/evaluate_model.py")
        
    else:
        print("❌ No models evaluated successfully")
    
    return results

if __name__ == "__main__":
    results = quick_evaluate()