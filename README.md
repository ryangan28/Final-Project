# 🌱 Organic Farm Pest Management AI System

A streamlined, intelligent AI system for organic farmers that identifies agricultural pests through computer vision and provides OMRI-certified treatment recommendations with an intuitive 3-page interface.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=flat&logo=Streamlit&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/ryangan28/Final-Project.git
cd Final-Project

# Smart launcher handles dependencies automatically
python start.py
```

The application will automatically launch at `http://localhost:8501`

## ✨ Key Features

### 🔍 Advanced Pest Detection
- **EfficientNet-B0 Deep Learning**: High accuracy with ensemble of 5 models and uncertainty quantification
- **YOLOv8-nano Integration**: Fast object detection with pest-specific training
- **Multi-Backend System**: Graceful degradation from ML to simulation modes
- **12 Pest Classes**: Comprehensive coverage of common agricultural pests
- **Real-time Processing**: Optimized inference with CPU/GPU support
- **Model Selection**: Choose between different detection models for speed vs accuracy

### 💬 Conversational AI Assistant
- **LM Studio Integration**: Local LLM for privacy-preserving conversations
- **Context-Aware Responses**: Integrates pest detection results automatically
- **Agricultural Expertise**: Specialized prompts for farming scenarios
- **Offline Capability**: Works without internet connection
- **Quick Actions**: Pre-defined prompts for common farming questions

### 🌿 Organic Treatment Library
- **OMRI-Certified Solutions**: 100% organic-compliant treatments
- **IPM Principles**: Integrated Pest Management approach
- **Severity-Based Plans**: Scaled responses (low/medium/high)
- **Comprehensive Coverage**: Mechanical, biological, and cultural controls
- **Smart Navigation**: Context-aware library showing relevant treatments
- **Pest-Specific Mapping**: Automatic pest-to-treatment matching

### 📱 Streamlined User Interface
- **3-Page Design**: Focused workflow with Pest Identification, Chat Assistant, and Treatment Library
- **Mobile-Friendly**: Responsive design optimized for field use
- **Context-Aware Navigation**: Seamlessly transition between identification and treatments
- **Real-time Results**: Instant pest identification with confidence scores
- **Smart Treatment Selection**: Pre-selects relevant treatments based on pest identification

## 🐛 Supported Pest Types

| Pest | Scientific Name | Type | Treatment Mapping |
|------|----------------|------|-------------------|
| Ants | Formicidae | Mixed Impact | → Aphids (General Organic) |
| Bees | Apis mellifera | **Beneficial** ✅ | → Aphids (Bee-Friendly) |
| Beetles | Coleoptera | Crop Pest | → Colorado Potato Beetle |
| Caterpillars | Lepidoptera larvae | Crop Pest | → Caterpillars |
| Earthworms | Oligochaeta | **Beneficial** ✅ | → Aphids (Beneficial Preservation) |
| Earwigs | Dermaptera | Mixed Impact | → Flea Beetle |
| Grasshoppers | Orthoptera | Crop Pest | → Caterpillars |
| Moths | Lepidoptera | Crop Pest | → Caterpillars |
| Slugs | Gastropoda | Crop Pest | → Aphids (General Organic) |
| Snails | Gastropoda | Crop Pest | → Aphids (General Organic) |
| Wasps | Hymenoptera | Mixed Impact | → Aphids (Beneficial-Friendly) |
| Weevils | Curculionidae | Crop Pest | → Cucumber Beetle |

## 🏗️ Architecture

```
Final-Project/
├── 📱 main.py                    # Main application orchestrator
├── 🚀 start.py                   # Smart launcher with dependency management
├── 📋 requirements.txt           # Python dependencies (51 lines total)
│
├── 👁️ vision/                    # Computer Vision Module (805 lines)
│   └── pest_detector.py          # Unified detector with multiple backends
│
├── 🌿 treatments/                # Treatment Recommendation Engine (426 lines)
│   └── recommendation_engine.py  # Organic treatment database & IPM logic
│
├── 💬 conversation/              # Conversational AI Module
│   ├── chat_interface.py         # Chat UI and response generation (402 lines)
│   └── llm_integration.py        # LM Studio API integration
│
├── 📱 mobile/                    # Streamlined Web Interface (807 lines)
│   └── app_interface.py          # 3-page Streamlit application
│                                 # • Pest Identification
│                                 # • Chat Assistant  
│                                 # • Treatment Library
│
├── ⚡ edge/                      # Edge Optimization
│   └── model_optimizer.py        # Model compression for deployment
│
├── 🧠 models/                    # Trained Models
│   ├── archive/                  # Archived model versions
│   ├── efficientnet/            # EfficientNet-B0 ensemble models
│   │   └── v3/                   # Latest version (93.17% accuracy)
│   ├── optimized/                # Edge-optimized versions
│   ├── yolo/                     # YOLOv8 model variants
│   └── optimized_training.log    # Training optimization logs
│
├── 🎓 training/                  # Training Pipeline
│   ├── scripts/                  # Training scripts directory
│   │   ├── efficientnet_train.py    # Full EfficientNet training script
│   │   ├── efficientnet_quick_train.py # Fast EfficientNet training
│   │   ├── yolo_train.py            # YOLOv8 training pipeline
│   │   ├── yolo_quick_train.py      # Fast YOLOv8 training
│   │   └── evaluate_models.py       # Model evaluation suite
│   ├── notebooks/                # Training notebooks (Jupyter)
│   ├── logs/                     # Training logs
│   ├── datasets_split/           # Training data splits
│   ├── README.md                 # Training documentation
│   └── REORGANIZATION_SUMMARY.md # Training structure notes
│
├── 🗂️ datasets/                  # Pest Image Dataset (5,494 total images)
│   ├── ants/                     # Ant species images (499 images)
│   ├── bees/                     # Bee species images (500 images)
│   ├── beetle/                   # Beetle species images (416 images)
│   ├── catterpillar/             # Caterpillar images (434 images)
│   ├── earthworms/               # Earthworm images (323 images)
│   ├── earwig/                   # Earwig images (466 images)
│   ├── grasshopper/              # Grasshopper images (485 images)
│   ├── moth/                     # Moth species images (497 images)
│   ├── slug/                     # Slug images (391 images)
│   ├── snail/                    # Snail images (500 images)
│   ├── wasp/                     # Wasp species images (498 images)
│   └── weevil/                   # Weevil images (485 images)
│
├── 📚 docs/                      # Documentation
│   └── Final Project Topic - Organic Farm Pest Management AI System.md
│
├── 🌍 locales/                   # Internationalization
│   └── en.json                   # English language strings
│
├── 📝 logs/                      # Application Logs
│   └── pest_management.log       # System operation logs
│
├── 🧪 tests/                     # Test Suite
│   └── test_system.py            # Comprehensive system tests
│
└── 📁 config/                    # Configuration files
```

## 🔧 Technical Requirements

### System Requirements
- **Python**: 3.8 or higher (verified by smart launcher)
- **Operating System**: Windows, macOS, Linux
- **Memory**: 4GB RAM minimum (8GB recommended for ML features)
- **Storage**: 2GB free space for models and datasets
- **GPU**: Optional (CUDA-compatible for enhanced performance)
- **Internet**: Not required for core functionality (offline-first design)

### Core Dependencies (Required)
```
streamlit>=1.28.0          # Streamlined 3-page web interface
Pillow>=9.5.0              # Image processing for pest identification
numpy>=1.24.0              # Numerical computing
```

### Enhanced ML Stack (Optional - for full functionality)
```
torch>=2.0.0               # PyTorch deep learning framework
torchvision>=0.15.0        # Computer vision models and utilities
ultralytics>=8.0.0         # YOLO models for object detection
scikit-learn>=1.3.0        # Machine learning utilities for fallback modes
python-dotenv>=1.0.0       # Environment variable management
requests>=2.31.0           # HTTP client for LM Studio API integration
matplotlib>=3.7.0          # Plotting and model visualization
seaborn>=0.12.0           # Statistical visualization
pandas>=2.0.0             # Data manipulation and analysis
```

### Edge Optimization (Optional)
```
onnx>=1.14.0               # Model optimization for edge deployment
psutil>=5.9.0              # System performance monitoring
```

## 🔧 Technical Details

### Machine Learning Stack
- **Primary Architecture**: EfficientNet-B0 with custom classification head and ensemble methodology
- **Secondary Detection**: YOLOv8-nano for fast object detection and classification  
- **Training Strategy**: 5-fold cross-validation with stratified splits and agricultural-specific augmentations
- **Uncertainty Estimation**: Monte Carlo Dropout with temperature scaling for confidence assessment
- **Fallback System**: Multi-tiered detection with graceful degradation capabilities

### System Capabilities
- **Offline-First**: Full functionality without internet connection
- **Multi-Backend Detection**: EfficientNet ensemble → YOLOv8 → Basic ML → Simulation fallback
- **Lightweight Mode**: CPU-only operation with reduced dependencies
- **Enhanced Mode**: GPU acceleration with full ML stack
- **Smart Launcher**: Automatic dependency checking and environment setup

### Interface Innovations
- **Context-Aware Navigation**: Treatment library pre-selects pests based on identification results
- **Smart Pest Mapping**: Automatically maps detected pests to available treatment categories
- **Seamless Workflow**: One-click transition from identification to specific treatments
- **Simplified Design**: Focused 3-page interface reduces cognitive load for farmers

## 💻 Installation & Quick Start

### Method 1: Quick Start (Lightweight - Basic Features)
```bash
# Clone the repository
git clone https://github.com/ryangan28/Final-Project.git
cd Final-Project

# Install core dependencies only
pip install streamlit pillow numpy

# Launch with basic features
python start.py
```

### Method 2: Full Installation (Complete ML Stack)
```bash
# Clone the repository
git clone https://github.com/ryangan28/Final-Project.git
cd Final-Project

# Install all dependencies
pip install -r requirements.txt

# Launch with all features
python start.py
```

### Method 3: Smart Launcher (Recommended)
```bash
# The smart launcher automatically checks dependencies and installs missing ones
python start.py --setup
```

The application will automatically:
- Check Python version compatibility (requires 3.8+)
- Verify and install missing dependencies
- Launch the web interface at `http://localhost:8501`
- Gracefully degrade features based on available dependencies

## 🗂️ Dataset Information

The system uses a comprehensive agricultural pest dataset with **5,494 total images** organized into 12 categories:

- **ants** - Garden ants, worker ants, fire ants (499 images)
- **bees** - Honey bees, worker bees, bumble bees *(beneficial)* (500 images)
- **beetle** - Flea beetles, Japanese beetles, Colorado potato beetles (416 images)
- **catterpillar** - Hornworms, cabbage worms, armyworms (434 images) *(Note: folder name maintains original spelling)*
- **earthworms** - Garden earthworms, red worms *(beneficial)* (323 images)
- **earwig** - European earwigs, common earwigs (466 images)
- **grasshopper** - Locusts, field grasshoppers (485 images)
- **moth** - Codling moths, cabbage moths, corn borers (497 images)
- **slug** - Garden slugs, gray field slugs (391 images)
- **snail** - Garden snails, brown garden snails (500 images)
- **wasp** - Paper wasps, yellow jackets, parasitic wasps (498 images)
- **weevil** - Boll weevils, rice weevils, grain weevils (485 images)

**Dataset Characteristics:**
- Classes are reasonably balanced (ratio: 1.5)
- High-quality images suitable for production training
- Real-world farming scenarios represented
- Compatible with both EfficientNet and YOLO training pipelines

## 🎯 Usage Guide

### 1. Pest Identification
1. Launch the application: `python start.py`
2. Navigate to "🔍 Pest Identification" (default page)
3. Select your preferred detection model from the dropdown
4. Upload a clear photo of the pest or crop damage
5. Click "🔬 Analyze Image" for instant results
6. View confidence scores, severity assessment, and detection method

### 2. Treatment Recommendations
- **Automatic Flow**: Click "📚 View Treatment Library" after pest identification
- **Smart Pre-selection**: System automatically selects the closest matching pest
- **Context Display**: See which pest was originally identified vs library match
- **Quick Summary**: Immediate action recommendations before detailed treatments
- **Browse Mode**: Switch to manual browsing with "🔄 Browse All Pests"

### 3. Chat Assistant
- **Context-Aware**: Click "💬 Chat About Treatment" for pest-specific guidance
- **Natural Language**: Ask questions about organic farming and pest management
- **Quick Actions**: Use pre-defined buttons for common farming scenarios
- **Integrated Responses**: AI considers your recent pest identification results

## 🧪 Training Your Own Models

### Full Training (Production)  
```bash
# Complete EfficientNet ensemble training with 5-fold cross-validation
python training/scripts/efficientnet_train.py

# Results: 93.17% ± 1.32% accuracy across folds
```

### Quick Training (Development/Testing)
```bash
# Fast training for development and testing
python training/scripts/efficientnet_quick_train.py
```

### YOLOv8 Training
```bash
python training/scripts/yolo_train.py
```

### Evaluate Models
```bash
python training/scripts/evaluate_models.py
```

## 📊 Performance Metrics

Based on comprehensive training and evaluation:

| Backend | Accuracy | Inference Time | Model Size | Use Case |
|---------|----------|----------------|------------|----------|
| EfficientNet Ensemble (v3) | **93.17% ± 1.32%** | ~200-500ms | ~85MB (5 models) | Production |
| YOLOv8-nano | Good (varies) | ~50-100ms | ~6MB | Edge/Mobile |
| Basic ML Fallback | Moderate | ~10-50ms | Minimal | Emergency |
| Simulation Mode | N/A | <10ms | None | Testing |

**EfficientNet v3 Detailed Results:**
- **Cross-Validation**: 5-fold stratified validation
- **Best Fold**: 95.3% accuracy
- **Worst Fold**: 91.3% accuracy  
- **Consistency**: Very stable across different data splits
- **Status**: Production-ready for commercial deployment

*Note: Performance metrics vary based on hardware and image complexity*

## 🔄 User Workflow

### Streamlined 3-Page Experience

```mermaid
graph TD
    A[🔍 Pest Identification] --> B[Upload Image]
    B --> C[Select Model]
    C --> D[Analyze]
    D --> E{Results}
    E --> F[💬 Chat Assistant]
    E --> G[📚 Treatment Library]
    F --> H[Context-Aware Chat]
    G --> I[Smart Pest Selection]
    I --> J[Quick Treatment Summary]
    J --> K[Detailed Treatments]
    H --> L[Continue Conversation]
    K --> M[Apply Treatments]
```

### Key Interface Features
- **Single-Page Focus**: Each page serves a specific purpose without confusion
- **Context Preservation**: Your pest identification follows you across pages
- **Smart Defaults**: System pre-selects the most relevant options
- **Clear Navigation**: Simple sidebar with just the 3 essential functions

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
    'pest_type': 'beetle',
    'confidence': 0.92,
    'uncertainty': 0.05,
    'backend_used': 'efficientnet_ensemble',
    'metadata': {
        'detection_time': 0.234,
        'model_version': 'improved_fold_0',
        'image_size': [224, 224]
    }
}
```

### Treatment Library Context API
```python
# Store pest context for treatment library
st.session_state.library_pest_context = 'ants'

# Treatment library automatically maps to closest match
# 'ants' → 'Aphids' (general organic treatments)
# 'beetle' → 'Colorado Potato Beetle'
# 'caterpillars' → 'Caterpillars'
```

## 👥 Development Team

**Singapore Institute of Technology (SIT)**  
Overseas Immersion Programme - Final Project

**Team Members:**
- **Ryan Koo Wei Feng** - Information Security (IS)
- **Farihin Fatten Binte Abdul Rahman** - Information Security (IS)
- **Khoo Ye Chen** - Software Engineering (SE)
- **Gan Kang Ting, Ryan** - Information Security (IS)
- **Donovan Leong Jia Le** - Applied Artificial Intelligence (AI)

**Academic Collaboration:**
- 🏫 **Home Institution**: Singapore Institute of Technology (SIT)
- 🌏 **Host Institution**: FPT University Da Nang, Vietnam

## 🚀 Project Innovations

### Interface Design
- **Reduced Complexity**: 3 focused workflows
- **Context Preservation**: Pest identification results seamlessly flow to treatment recommendations
- **Smart Mapping**: Automatic pest-to-treatment category matching reduces user confusion
- **Mobile-First**: Optimized for field use with simplified navigation

### Technical Achievements
- **Multi-Model Selection**: Real-time switching between detection models
- **Graceful Degradation**: System works even with minimal dependencies
- **Context-Aware AI**: Chat assistant understands your specific pest situation
- **Edge Optimization**: Lightweight deployment suitable for resource-constrained environments

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
- **Academic Supervisors**: SIT and FPT University mentors

## 📞 Support & Documentation

- **Issues**: Please use the [GitHub Issues page](https://github.com/ryangan28/Final-Project/issues)
- **Project Documentation**: Check the `docs/` directory for detailed project information
- **Training Documentation**: See `training/README.md` for model training details
- **System Logs**: Review `logs/pest_management.log` for troubleshooting and system monitoring
- **Dataset Requirements**: Ensure the `datasets/` folder contains all 12 pest categories before training

### Troubleshooting

**Common Issues:**
- **Missing Dependencies**: Run `python start.py --setup` to install missing packages
- **Model Loading Errors**: System will gracefully degrade to fallback detection modes
- **Port Conflicts**: If port 8501 is busy, Streamlit will automatically suggest an alternative
- **Memory Issues**: Use lightweight mode by installing only core dependencies

**System Capabilities by Installation Type:**
- **Lightweight**: Basic pest identification using simulation mode
- **Full ML Stack**: Complete EfficientNet + YOLO detection with chat assistant
- **Development**: All features plus training and evaluation capabilities

---

*🌱 Supporting sustainable agriculture through streamlined AI-powered pest management*
