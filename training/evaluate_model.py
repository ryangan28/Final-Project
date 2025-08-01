"""
Comprehensive Model Evaluation and Testing Suite
==============================================

Evaluate trained models with detailed metrics, uncertainty analysis,
and production readiness assessment.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import ML dependencies
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from sklearn.metrics import (
        classification_report, confusion_matrix, 
        accuracy_score, precision_recall_fscore_support,
        roc_auc_score, average_precision_score
    )
    from sklearn.calibration import calibration_curve
    import pandas as pd
    ML_AVAILABLE = True
    logger.info("Full ML dependencies loaded successfully")
except ImportError as e:
    ML_AVAILABLE = False
    logger.error(f"ML dependencies not available: {e}")
    print("Please install: pip install torch torchvision matplotlib seaborn scikit-learn pandas")
    sys.exit(1)

# Import our improved components
sys.path.append(str(Path(__file__).parent.parent))
from training.improved_train import ImprovedPestDataset, AgriculturalAugmentations
from vision.improved_pest_detector import ImprovedPestDetector, EfficientNetPestClassifier


class ModelEvaluator:
    """Comprehensive model evaluation suite."""
    
    def __init__(self, model_dir: str = "models/improved", data_dir: str = "datasets"):
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.results_dir = self.model_dir / "evaluation_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load class mapping
        self.class_mapping = self._load_class_mapping()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Evaluating models on device: {self.device}")
    
    def _load_class_mapping(self) -> Dict:
        """Load class mapping from training."""
        class_mapping_path = self.model_dir / 'class_mapping.json'
        if class_mapping_path.exists():
            with open(class_mapping_path, 'r') as f:
                mapping = json.load(f)
                # Ensure the mapping has the expected structure
                if isinstance(mapping.get('classes'), list) and isinstance(mapping.get('class_to_idx'), dict):
                    return mapping
                else:
                    logger.warning(f"Invalid class mapping format in {class_mapping_path}")
        
        # Default mapping - try multiple model directories
        for model_dir in [self.model_dir, Path("models/improved_quick"), Path("models/improved")]:
            class_mapping_path = model_dir / 'class_mapping.json'
            if class_mapping_path.exists():
                try:
                    with open(class_mapping_path, 'r') as f:
                        mapping = json.load(f)
                        if isinstance(mapping.get('classes'), list) and isinstance(mapping.get('class_to_idx'), dict):
                            logger.info(f"Loaded class mapping from {class_mapping_path}")
                            return mapping
                except Exception as e:
                    logger.warning(f"Failed to load class mapping from {class_mapping_path}: {e}")
        
        # Final fallback
        logger.warning("Using default class mapping")
        from vision.improved_pest_detector import ImprovedPestDetector
        detector = ImprovedPestDetector()
        return {
            'classes': list(detector.PEST_INFO.keys()),
            'class_to_idx': {cls: idx for idx, cls in enumerate(detector.PEST_INFO.keys())},
            'num_classes': len(detector.PEST_INFO)
        }
    
    def evaluate_all_models(self) -> Dict:
        """Evaluate all trained models."""
        logger.info("Starting comprehensive model evaluation...")
        
        # Find all model files
        model_paths = list(self.model_dir.glob('best_model_fold_*.pth'))
        if not model_paths:
            logger.error("No trained models found!")
            return {}
        
        # Prepare test dataset
        test_dataset = self._prepare_test_dataset()
        if len(test_dataset) == 0:
            logger.error("No test data available!")
            return {}
        
        # Evaluate each model
        model_results = {}
        for model_path in model_paths:
            fold_num = self._extract_fold_number(model_path)
            logger.info(f"Evaluating model fold {fold_num}...")
            
            try:
                results = self.evaluate_single_model(model_path, test_dataset)
                model_results[f'fold_{fold_num}'] = results
            except Exception as e:
                logger.error(f"Failed to evaluate model {model_path}: {e}")
                model_results[f'fold_{fold_num}'] = {'error': str(e)}
        
        # Evaluate ensemble
        if len(model_results) > 1:
            logger.info("Evaluating ensemble performance...")
            ensemble_results = self.evaluate_ensemble(model_paths, test_dataset)
            model_results['ensemble'] = ensemble_results
        
        # Generate comprehensive report
        final_report = self._generate_comprehensive_report(model_results, test_dataset)
        
        # Save results
        self._save_evaluation_results(final_report)
        
        return final_report
    
    def _prepare_test_dataset(self):
        """Prepare test dataset for evaluation."""
        augmentations = AgriculturalAugmentations()
        
        # Create test dataset with proper class mapping
        test_dataset = ImprovedPestDataset(
            self.data_dir,
            transform=augmentations.get_val_transforms(),
            class_mapping=self.class_mapping
        )
        
        if len(test_dataset.samples) == 0:
            logger.error(f"No samples found in dataset directory: {self.data_dir}")
            return test_dataset
        
        # Use 20% of data for testing (simulate test split)
        total_samples = len(test_dataset.samples)
        if total_samples > 10:  # Only subsample if we have enough data
            test_size = max(10, int(0.2 * total_samples))  # At least 10 samples
            np.random.seed(42)  # For reproducibility
            test_indices = np.random.choice(
                total_samples, 
                size=min(test_size, total_samples), 
                replace=False
            )
            test_samples = [test_dataset.samples[i] for i in test_indices]
            test_dataset.samples = test_samples
        
        logger.info(f"Test dataset prepared: {len(test_dataset)} samples from {total_samples} total")
        return test_dataset
    
    def _extract_fold_number(self, model_path: Path) -> int:
        """Extract fold number from model filename."""
        try:
            return int(str(model_path.stem).split('_')[-1])
        except:
            return 0
    
    def evaluate_single_model(self, model_path: Path, test_dataset) -> Dict:
        """Evaluate a single model."""
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        model = EfficientNetPestClassifier(num_classes=self.class_mapping['num_classes'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        # Create data loader
        test_loader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        # Collect predictions
        all_predictions = []
        all_labels = []
        all_probabilities = []
        inference_times = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Measure inference time
                start_time = time.time()
                outputs = model(images)
                inference_time = (time.time() - start_time) * 1000  # ms
                inference_times.append(inference_time)
                
                # Get predictions
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        results = self._calculate_detailed_metrics(
            all_labels, all_predictions, all_probabilities, inference_times
        )
        
        # Add model-specific info
        results['model_path'] = str(model_path)
        results['training_accuracy'] = checkpoint.get('val_acc', 0.0)
        results['training_epoch'] = checkpoint.get('epoch', 0)
        
        return results
    
    def evaluate_ensemble(self, model_paths: List[Path], test_dataset) -> Dict:
        """Evaluate ensemble of models."""
        # Load all models
        models = []
        weights = []
        
        for model_path in model_paths:
            checkpoint = torch.load(model_path, map_location=self.device)
            model = EfficientNetPestClassifier(num_classes=self.class_mapping['num_classes'])
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            models.append(model)
            weights.append(checkpoint.get('val_acc', 1.0))  # Weight by validation accuracy
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Create data loader
        test_loader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        # Collect ensemble predictions
        all_predictions = []
        all_labels = []
        all_probabilities = []
        inference_times = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Measure inference time
                start_time = time.time()
                
                # Get predictions from all models
                ensemble_outputs = []
                for model in models:
                    outputs = model(images)
                    probabilities = F.softmax(outputs, dim=1)
                    ensemble_outputs.append(probabilities)
                
                # Weighted ensemble
                ensemble_probs = sum(w * probs for w, probs in zip(weights, ensemble_outputs))
                _, predicted = torch.max(ensemble_probs, 1)
                
                inference_time = (time.time() - start_time) * 1000  # ms
                inference_times.append(inference_time)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(ensemble_probs.cpu().numpy())
        
        # Calculate metrics
        results = self._calculate_detailed_metrics(
            all_labels, all_predictions, all_probabilities, inference_times
        )
        
        # Add ensemble-specific info
        results['num_models'] = len(models)
        results['model_weights'] = weights.tolist()
        results['ensemble_method'] = 'weighted_average'
        
        return results
    
    def _calculate_detailed_metrics(self, labels, predictions, probabilities, inference_times) -> Dict:
        """Calculate comprehensive evaluation metrics."""
        # Convert to numpy arrays
        labels = np.array(labels)
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        
        # Basic metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )
        
        # Overall averages
        macro_precision = precision.mean()
        macro_recall = recall.mean()
        macro_f1 = f1.mean()
        
        # Weighted averages
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        per_class_metrics = {}
        for i, class_name in enumerate(self.class_mapping['classes']):
            per_class_metrics[class_name] = {
                'precision': float(precision[i]) if i < len(precision) else 0.0,
                'recall': float(recall[i]) if i < len(recall) else 0.0,
                'f1_score': float(f1[i]) if i < len(f1) else 0.0,
                'support': int(support[i]) if i < len(support) else 0
            }
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        # Confidence statistics
        max_probabilities = np.max(probabilities, axis=1)
        confidence_stats = {
            'mean_confidence': float(np.mean(max_probabilities)),
            'std_confidence': float(np.std(max_probabilities)),
            'min_confidence': float(np.min(max_probabilities)),
            'max_confidence': float(np.max(max_probabilities))
        }
        
        # Calibration analysis
        calibration_results = self._analyze_calibration(labels, predictions, probabilities)
        
        # Performance metrics
        performance_stats = {
            'mean_inference_time_ms': float(np.mean(inference_times)),
            'std_inference_time_ms': float(np.std(inference_times)),
            'throughput_samples_per_second': float(1000 / np.mean(inference_times))
        }
        
        # Model size (if available)
        total_params = sum(p.numel() for p in EfficientNetPestClassifier(
            self.class_mapping['num_classes']
        ).parameters())
        
        return {
            'accuracy': float(accuracy),
            'macro_precision': float(macro_precision),
            'macro_recall': float(macro_recall),
            'macro_f1': float(macro_f1),
            'weighted_precision': float(weighted_precision),
            'weighted_recall': float(weighted_recall),
            'weighted_f1': float(weighted_f1),
            'per_class_metrics': per_class_metrics,
            'confusion_matrix': cm.tolist(),
            'confidence_statistics': confidence_stats,
            'calibration_analysis': calibration_results,
            'performance_statistics': performance_stats,
            'model_parameters': total_params,
            'num_test_samples': len(labels)
        }
    
    def _analyze_calibration(self, labels, predictions, probabilities) -> Dict:
        """Analyze model calibration."""
        try:
            # Get maximum probabilities and check if prediction is correct
            max_probs = np.max(probabilities, axis=1)
            correct_predictions = (predictions == labels)
            
            # Expected Calibration Error (ECE)
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0
            bin_accuracies = []
            bin_confidences = []
            bin_counts = []
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (max_probs > bin_lower) & (max_probs <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = correct_predictions[in_bin].mean()
                    avg_confidence_in_bin = max_probs[in_bin].mean()
                    
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                    
                    bin_accuracies.append(float(accuracy_in_bin))
                    bin_confidences.append(float(avg_confidence_in_bin))
                    bin_counts.append(int(in_bin.sum()))
                else:
                    bin_accuracies.append(0.0)
                    bin_confidences.append(0.0)
                    bin_counts.append(0)
            
            return {
                'expected_calibration_error': float(ece),
                'bin_accuracies': bin_accuracies,
                'bin_confidences': bin_confidences,
                'bin_counts': bin_counts,
                'reliability_diagram_data': {
                    'bin_boundaries': bin_boundaries.tolist(),
                    'accuracies': bin_accuracies,
                    'confidences': bin_confidences
                }
            }
            
        except Exception as e:
            logger.warning(f"Calibration analysis failed: {e}")
            return {'expected_calibration_error': 0.0, 'error': str(e)}
    
    def _generate_comprehensive_report(self, model_results: Dict, test_dataset) -> Dict:
        """Generate comprehensive evaluation report."""
        # Extract key metrics from all models
        model_accuracies = []
        model_f1_scores = []
        model_inference_times = []
        
        for model_name, results in model_results.items():
            if 'error' not in results:
                model_accuracies.append(results['accuracy'])
                model_f1_scores.append(results['weighted_f1'])
                model_inference_times.append(results['performance_statistics']['mean_inference_time_ms'])
        
        # Calculate statistics across models
        if model_accuracies:
            cross_model_stats = {
                'mean_accuracy': float(np.mean(model_accuracies)),
                'std_accuracy': float(np.std(model_accuracies)),
                'best_accuracy': float(np.max(model_accuracies)),
                'worst_accuracy': float(np.min(model_accuracies)),
                'mean_f1_score': float(np.mean(model_f1_scores)),
                'std_f1_score': float(np.std(model_f1_scores)),
                'mean_inference_time_ms': float(np.mean(model_inference_times)),
                'std_inference_time_ms': float(np.std(model_inference_times))
            }
        else:
            cross_model_stats = {}
        
        # Analyze class performance across models
        class_performance_analysis = self._analyze_class_performance(model_results)
        
        # Production readiness assessment
        production_assessment = self._assess_production_readiness(model_results, cross_model_stats)
        
        return {
            'evaluation_timestamp': time.time(),
            'test_dataset_size': len(test_dataset),
            'num_classes': self.class_mapping['num_classes'],
            'class_names': self.class_mapping['classes'],
            'individual_model_results': model_results,
            'cross_model_statistics': cross_model_stats,
            'class_performance_analysis': class_performance_analysis,
            'production_readiness_assessment': production_assessment,
            'recommendations': self._generate_recommendations(model_results, cross_model_stats)
        }
    
    def _analyze_class_performance(self, model_results: Dict) -> Dict:
        """Analyze performance across different pest classes."""
        class_stats = defaultdict(list)
        
        # Collect per-class metrics from all models
        for model_name, results in model_results.items():
            if 'error' not in results and 'per_class_metrics' in results:
                for class_name, metrics in results['per_class_metrics'].items():
                    class_stats[class_name].append(metrics)
        
        # Calculate statistics for each class
        class_analysis = {}
        for class_name, metrics_list in class_stats.items():
            if metrics_list:
                f1_scores = [m['f1_score'] for m in metrics_list]
                precisions = [m['precision'] for m in metrics_list]
                recalls = [m['recall'] for m in metrics_list]
                supports = [m['support'] for m in metrics_list]
                
                class_analysis[class_name] = {
                    'mean_f1_score': float(np.mean(f1_scores)),
                    'std_f1_score': float(np.std(f1_scores)),
                    'mean_precision': float(np.mean(precisions)),
                    'mean_recall': float(np.mean(recalls)),
                    'avg_support': float(np.mean(supports)),
                    'difficulty_ranking': 1.0 - np.mean(f1_scores)  # Higher = more difficult
                }
        
        # Identify challenging classes
        if class_analysis:
            difficulty_scores = [(name, data['difficulty_ranking']) 
                               for name, data in class_analysis.items()]
            difficulty_scores.sort(key=lambda x: x[1], reverse=True)
            
            challenging_classes = [name for name, _ in difficulty_scores[:3]]
            easy_classes = [name for name, _ in difficulty_scores[-3:]]
        else:
            challenging_classes = []
            easy_classes = []
        
        return {
            'per_class_statistics': class_analysis,
            'most_challenging_classes': challenging_classes,
            'easiest_classes': easy_classes
        }
    
    def _assess_production_readiness(self, model_results: Dict, cross_model_stats: Dict) -> Dict:
        """Assess production readiness of models."""
        # Define production criteria
        production_criteria = {
            'min_accuracy': 0.80,  # 80% minimum accuracy
            'max_inference_time_ms': 500,  # 500ms max inference time
            'min_confidence_reliability': 0.85,  # Well-calibrated confidence
            'max_model_size_mb': 100,  # 100MB max model size
            'min_cross_validation_stability': 0.05  # Max 5% std in CV accuracy
        }
        
        # Assess each criterion
        assessment = {}
        
        if cross_model_stats:
            # Accuracy criterion
            best_accuracy = cross_model_stats.get('best_accuracy', 0.0)
            assessment['accuracy_ready'] = best_accuracy >= production_criteria['min_accuracy']
            assessment['accuracy_score'] = best_accuracy
            
            # Inference time criterion
            mean_inference_time = cross_model_stats.get('mean_inference_time_ms', float('inf'))
            assessment['speed_ready'] = mean_inference_time <= production_criteria['max_inference_time_ms']
            assessment['inference_time_ms'] = mean_inference_time
            
            # Stability criterion
            accuracy_std = cross_model_stats.get('std_accuracy', float('inf'))
            assessment['stability_ready'] = accuracy_std <= production_criteria['min_cross_validation_stability']
            assessment['accuracy_std'] = accuracy_std
        else:
            assessment = {
                'accuracy_ready': False,
                'speed_ready': False,
                'stability_ready': False,
                'error': 'No valid model results available'
            }
        
        # Overall readiness
        readiness_checks = [
            assessment.get('accuracy_ready', False),
            assessment.get('speed_ready', False),
            assessment.get('stability_ready', False)
        ]
        
        assessment['overall_production_ready'] = all(readiness_checks)
        assessment['readiness_score'] = sum(readiness_checks) / len(readiness_checks)
        assessment['production_criteria'] = production_criteria
        
        return assessment
    
    def _generate_recommendations(self, model_results: Dict, cross_model_stats: Dict) -> List[str]:
        """Generate actionable recommendations based on evaluation results."""
        recommendations = []
        
        if not cross_model_stats:
            recommendations.append("❌ No valid models found - retrain with proper data")
            return recommendations
        
        # Accuracy recommendations
        best_accuracy = cross_model_stats.get('best_accuracy', 0.0)
        if best_accuracy < 0.80:
            recommendations.append(f"🎯 Improve accuracy ({best_accuracy:.2%}) with more training data or better augmentation")
        elif best_accuracy > 0.90:
            recommendations.append(f"✅ Excellent accuracy ({best_accuracy:.2%}) - ready for production")
        
        # Inference time recommendations
        mean_time = cross_model_stats.get('mean_inference_time_ms', 0)
        if mean_time > 500:
            recommendations.append(f"⚡ Optimize inference speed ({mean_time:.1f}ms) with model compression or hardware acceleration")
        elif mean_time < 100:
            recommendations.append(f"🚀 Excellent inference speed ({mean_time:.1f}ms) - suitable for real-time applications")
        
        # Stability recommendations
        accuracy_std = cross_model_stats.get('std_accuracy', 0.0)
        if accuracy_std > 0.05:
            recommendations.append(f"📊 Improve model stability (std: {accuracy_std:.3f}) with more consistent training or ensemble methods")
        
        # Ensemble recommendations
        if 'ensemble' in model_results and len(model_results) > 2:
            ensemble_acc = model_results['ensemble'].get('accuracy', 0.0)
            best_single_acc = max(r.get('accuracy', 0.0) for r in model_results.values() 
                                if 'accuracy' in r and r != model_results.get('ensemble', {}))
            
            if ensemble_acc > best_single_acc + 0.02:  # 2% improvement
                recommendations.append(f"🔄 Use ensemble method - {ensemble_acc:.2%} vs {best_single_acc:.2%} single model")
        
        # General recommendations
        recommendations.extend([
            "📈 Monitor model performance in production with A/B testing",
            "🔄 Implement gradual rollout with fallback to simulation mode",
            "📝 Set up automated retraining pipeline with new data",
            "🛡️ Add uncertainty thresholding for low-confidence predictions"
        ])
        
        return recommendations
    
    def _save_evaluation_results(self, report: Dict):
        """Save evaluation results to files."""
        # Save main report
        report_path = self.results_dir / 'evaluation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Evaluation report saved to: {report_path}")
        
        # Generate visualizations if matplotlib is available
        try:
            self._generate_visualizations(report)
        except Exception as e:
            logger.warning(f"Failed to generate visualizations: {e}")
    
    def _generate_visualizations(self, report: Dict):
        """Generate evaluation visualizations."""
        # Set style (compatible with different matplotlib/seaborn versions)
        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            try:
                plt.style.use('seaborn')
            except OSError:
                pass  # Use default style
        
        # 1. Accuracy comparison across models
        model_names = []
        accuracies = []
        
        for model_name, results in report['individual_model_results'].items():
            if 'accuracy' in results:
                model_names.append(model_name)
                accuracies.append(results['accuracy'])
        
        if model_names:
            plt.figure(figsize=(10, 6))
            bars = plt.bar(model_names, accuracies)
            plt.title('Model Accuracy Comparison')
            plt.ylabel('Accuracy')
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for bar, acc in zip(bars, accuracies):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{acc:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(self.results_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Per-class performance heatmap
        if 'class_performance_analysis' in report:
            class_data = report['class_performance_analysis']['per_class_statistics']
            if class_data:
                classes = list(class_data.keys())
                metrics = ['mean_precision', 'mean_recall', 'mean_f1_score']
                
                heatmap_data = []
                for metric in metrics:
                    row = [class_data[cls].get(metric, 0.0) for cls in classes]
                    heatmap_data.append(row)
                
                plt.figure(figsize=(12, 8))
                sns.heatmap(
                    heatmap_data,
                    xticklabels=classes,
                    yticklabels=metrics,
                    annot=True,
                    fmt='.3f',
                    cmap='RdYlBu_r',
                    center=0.5
                )
                plt.title('Per-Class Performance Heatmap')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(self.results_dir / 'class_performance_heatmap.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        logger.info(f"Visualizations saved to: {self.results_dir}")


def main():
    """Main evaluation function."""
    print("🧪 Comprehensive Model Evaluation Suite")
    print("=" * 50)
    
    # Check multiple possible model directories
    possible_dirs = [
        Path("models/improved_quick"),
        Path("models/improved"),
    ]
    
    model_dir = None
    model_files = []
    
    for directory in possible_dirs:
        if directory.exists():
            files = list(directory.glob('best_model_fold_*.pth'))
            if files:
                model_dir = directory
                model_files = files
                break
    
    if not model_dir or not model_files:
        print("❌ No trained models found!")
        print("   Searched directories:")
        for directory in possible_dirs:
            status = "✅ exists" if directory.exists() else "❌ not found"
            files_count = len(list(directory.glob('best_model_fold_*.pth'))) if directory.exists() else 0
            print(f"   {directory}: {status} ({files_count} models)")
        print()
        print("Please train models first using:")
        print("   python training/quick_improved_train.py  # Quick training")
        print("   python training/improved_train.py        # Full training")
        return
    
    print(f"📁 Found {len(model_files)} trained models in {model_dir}")
    
    # Initialize evaluator with the found model directory
    evaluator = ModelEvaluator(str(model_dir), data_dir="datasets")
    
    # Run evaluation
    start_time = time.time()
    results = evaluator.evaluate_all_models()
    evaluation_time = time.time() - start_time
    
    print(f"\n🎉 Evaluation completed in {evaluation_time:.2f} seconds!")
    
    # Print summary
    if 'cross_model_statistics' in results and results['cross_model_statistics']:
        stats = results['cross_model_statistics']
        print(f"\n📊 Summary Results:")
        print(f"   Best Accuracy: {stats.get('best_accuracy', 0.0):.2%}")
        print(f"   Mean Accuracy: {stats.get('mean_accuracy', 0.0):.2%} ± {stats.get('std_accuracy', 0.0):.3f}")
        print(f"   Mean F1-Score: {stats.get('mean_f1_score', 0.0):.3f}")
        print(f"   Mean Inference Time: {stats.get('mean_inference_time_ms', 0.0):.1f}ms")
        
        # Production readiness
        if 'production_readiness_assessment' in results:
            readiness = results['production_readiness_assessment']
            if readiness.get('overall_production_ready', False):
                print(f"✅ Models are production ready!")
            else:
                print(f"⚠️ Models need improvement for production (score: {readiness.get('readiness_score', 0.0):.1%})")
    
    # Print recommendations
    if 'recommendations' in results:
        print(f"\n💡 Recommendations:")
        for rec in results['recommendations'][:5]:  # Top 5 recommendations
            print(f"   {rec}")
    
    print(f"\n📁 Detailed results saved in: models/improved/evaluation_results/")
    
    return results


if __name__ == "__main__":
    if not ML_AVAILABLE:
        print("❌ ML dependencies not available. Please install required packages.")
        sys.exit(1)
    
    main()