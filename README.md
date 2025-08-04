# 🌱 Organic Farm Pest Management AI System

An intelligent, offline-first AI system for organic farmers that identifies agricultural pests through computer vision and provides OMRI-certified treatment recommendations.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=flat&logo=Streamlit&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/organic-pest-management-ai.git
cd organic-pest-management-ai

# Install dependencies
pip install -r requirements.txt

# Download the Agricultural Pests Dataset
# Place it in the 'datasets/' directory

# Launch the application
python start.py
```

The application will automatically check dependencies and launch at `http://localhost:8501`

## ✨ Key Features

### 🔍 Advanced Pest Detection
- **EfficientNet-B0 Deep Learning**: 93.1% accuracy with ensemble of 5 models
- **Uncertainty Quantification**: Monte Carlo Dropout for confidence estimation
- **12 Pest Classes**: Comprehensive coverage of common agricultural pests
- **Real-time Processing**: <200ms inference time per image

### 💬 Conversational AI Assistant
- **LM Studio Integration**: Local LLM for privacy-preserving conversations
- **Context-Aware Responses**: Integrates pest detection results
- **Agricultural Expertise**: Specialized prompts for farming scenarios
- **Offline Capability**: Works without internet connection

### 🌿 Organic Treatment Database
- **OMRI-Certified Solutions**: 100% organic-compliant treatments
- **IPM Principles**: Integrated Pest Management approach
- **Severity-Based Plans**: Scaled responses (low/medium/high)
- **Comprehensive Coverage**: Mechanical, biological, and cultural controls

### 📱 User Interface
- **Web Application**: Modern Streamlit interface
- **Mobile-Friendly**: Responsive design for field use
- **Image Upload**: Drag-and-drop or camera capture
- **Real-time Results**: Instant pest identification

## 🐛 Supported Pest Types

| Pest | Scientific Name | Type |
|------|----------------|------|
| Ants | Formicidae | Mixed Impact |
| Bees | Apis mellifera | **Beneficial** ✅ |
| Beetles | Coleoptera | Crop Pest |
| Caterpillars | Lepidoptera larvae | Crop Pest |
| Earthworms | Oligochaeta | **Beneficial** ✅ |
| Earwigs | Dermaptera | Mixed Impact |
| Grasshoppers | Orthoptera | Crop Pest |
| Moths | Lepidoptera | Crop Pest |
| Slugs | Gastropoda | Crop Pest |
| Snails | Gastropoda | Crop Pest |
| Wasps | Hymenoptera | Mixed Impact |
| Weevils | Curculionidae | Crop Pest |

## 🏗️ Architecture

```
organic-pest-management-ai/
├── 📱 main.py                    # Main application orchestrator
├── 🚀 start.py                   # Smart launcher with dependency management
├── 📋 requirements.txt           # Python dependencies
│
├── 👁️ vision/                    # Computer Vision Module
│   └── pest_detector.py          # Unified detector with multiple backends
│
├── 🌿 treatments/                # Treatment Recommendation Engine
│   └── recommendation_engine.py  # Organic treatment database & IPM logic
│
├── 💬 conversation/              # Conversational AI Module
│   ├── chat_interface.py         # Chat UI and response generation
│   └── llm_integration.py        # LM Studio API integration
│
├── 🌐 mobile/                    # Web Interface
│   └── app_interface.py          # Streamlit web application
│
├── ⚡ edge/                      # Edge Optimization
│   └── model_optimizer.py        # Model compression for deployment
│
├── 🧠 models/                    # Trained Models
│   ├── improved/                 # EfficientNet-B0 ensemble models
│   └── optimized/                # Edge-optimized versions
│
├── 🎓 training/                  # Training Pipeline
│   ├── improved_train.py         # EfficientNet training script
│   └── evaluate_model.py         # Model evaluation suite
│
└── 🧪 tests/                     # Test Suite
    └── test_system.py            # Comprehensive system tests
```

## 🔧 Technical Details

### Machine Learning Stack
- **Architecture**: EfficientNet-B0 with custom classification head
- **Training**: 5-fold cross-validation with stratified splits
- **Augmentations**: Agricultural-specific transformations
- **Uncertainty**: Monte Carlo Dropout + Temperature Scaling
- **Performance**: 93.1% accuracy (±0.57% std across folds)

### System Capabilities
- **Offline-First**: Full functionality without internet
- **Multi-Backend**: Graceful degradation across detection methods
- **Lightweight Mode**: Runs on CPU with reduced dependencies
- **Enhanced Mode**: GPU acceleration with full ML stack

## 💻 Installation Options

### Basic Installation (Lightweight)
```bash
pip install streamlit pillow numpy
python start.py
```

### Full Installation (ML-Enhanced)
```bash
pip install -r requirements.txt
python start.py --enhanced
```

### Development Installation
```bash
git clone <repository>
cd organic-pest-management-ai
pip install -r requirements.txt
python start.py --setup
```

## 🎯 Usage Guide

### 1. Pest Identification
1. Launch the application: `python start.py`
2. Navigate to "🔍 Pest Identification"
3. Upload a clear photo of the pest or crop damage
4. Click "Analyze" for instant results

### 2. Treatment Recommendations
- Automatic recommendations based on pest type and severity
- Browse the treatment library for detailed organic solutions
- Filter by treatment type: mechanical, biological, or cultural

### 3. Chat Assistant
- Ask questions in natural language
- Get context-aware responses based on detections
- Request specific guidance for your situation

## 🧪 Training Your Own Models

### Quick Training (Development)
```bash
python training/quick_improved_train.py
```

### Full Training (Production)
```bash
python training/improved_train.py
```

### Evaluate Models
```bash
python training/evaluate_model.py
```

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 93.1% ± 0.57% |
| Inference Time | <200ms |
| Model Size | ~17MB per model |
| Ensemble Size | 5 models |
| Parameters | 4.7M per model |

## 🌐 API Usage

### LM Studio Integration
The system integrates with LM Studio for conversational AI:

```python
# Automatic detection and connection to LM Studio
# Default endpoint: http://localhost:1234/v1
# Recommended model: llama-2-7b-chat
```

### Detection API
```python
from vision.pest_detector import UnifiedPestDetector

detector = UnifiedPestDetector()
result = detector.detect_pest("path/to/image.jpg")

# Result structure:
{
    'success': True,
    'pest_type': 'aphids',
    'confidence': 0.92,
    'uncertainty': 0.05,
    'metadata': {...}
}
```

## 👥 Development Team

**Singapore Institute of Technology (SIT)**  
Overseas Immersion Programme - Final Project

**Team Members:**
- Ryan Koo Wei Feng - Information Security (IS)
- Farihin Fatten Binte Abdul Rahman - Information Security (IS)
- Khoo Ye Chen - Software Engineering (SE)
- Gan Kang Ting, Ryan - Information Security (IS)
- Donovan Leong Jia Le - Applied Artificial Intelligence (AI)

**Academic Collaboration:**
- 🏫 Singapore Institute of Technology (SIT)
- 🌏 FPT University Da Nang

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/AmazingFeature`
3. Commit your changes: `git commit -m 'Add some AmazingFeature'`
4. Push to the branch: `git push origin feature/AmazingFeature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **EfficientNet**: Google Research for the efficient neural architecture
- **LM Studio**: For local LLM inference capabilities
- **Streamlit**: For the rapid web application framework
- **PyTorch**: For the deep learning framework
- **Agricultural Pests Dataset**: Kaggle dataset contributors
- **Organic Farming Community**: For domain expertise and testing

## 📞 Support

- **Issues**: Please use the GitHub Issues page
- **Documentation**: Check the `docs/` directory
- **Logs**: Review `logs/pest_management.log` for troubleshooting

---

*🌱 Supporting sustainable agriculture through AI-powered pest management*