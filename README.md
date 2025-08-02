# 🌱 Organic Farm Pest Management AI System

An intelligent, offline-first pest management system designed for organic farmers. Features computer vision pest identification using EfficientNet-B0 deep learning models, conversational AI powered by LM Studio, and comprehensive organic treatment recommend## 🙏 Acknowledgments

### **Academic Institutions**
- **Singapore Institute of Technology (SIT)** - Primary academic institution
- **FPT University Da Nang** - Overseas Immersion Programme partner
- **Academic supervisors and mentors** - Guidance and project oversight

### **Technical Foundations**
- **EfficientNet**: Efficient neural network architecture for computer vision
- **LM Studio**: Local LLM inference platform
- **Streamlit**: Rapid web application development framework
- **PyTorch**: Deep learning framework and ecosystem

### **Community Support**
- **Open-source community contributors** - Libraries, frameworks, and tools
- **Organic Farming Community** - Domain expertise and testing feedback
- **Agricultural extension services** - Real-world validation and requirementsthat maintain OMRI certification standards.

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+** (Required)
- **LM Studio** (Optional) - For enhanced conversational AI capabilities
  - Download from [https://lmstudio.ai/](https://lmstudio.ai/)

### Installation & Setup
```bash
# Clone the repository
git clone https://github.com/ryangan28/Final-Project.git
cd Final-Project

# Install dependencies
pip install -r requirements.txt

# Launch the application
python start.py
```

The application will automatically:
- Check dependencies and install missing packages
- Launch the Streamlit web interface
- Detect and integrate with LM Studio if available

### LM Studio Setup (Optional but Recommended)
1. Download and install [LM Studio](https://lmstudio.ai/)
2. Download a compatible model (recommended: Llama-2-7B-Chat-GGUF or similar)
3. Start the local server on `http://localhost:1234`
4. The system will automatically detect and integrate with LM Studio for enhanced chat capabilities

## ✨ Key Features

### 🔍 Advanced Pest Detection
- **EfficientNet-B0 Deep Learning Model** - Trained on 12 common farm pests
- **Uncertainty Quantification** - Monte Carlo Dropout + Temperature Scaling for confidence scoring
- **Graceful Degradation** - Multiple fallback modes ensure reliability
- **Real-time Processing** - Instant analysis of uploaded images

### 🤖 Intelligent Chat Assistant
- **LM Studio Integration** - Local LLM for private, offline conversations
- **Agricultural Domain Expertise** - Specialized prompts for farming scenarios
- **Context-Aware Responses** - Integrates pest identification results with treatment advice
- **Fallback Support** - Works with or without LM Studio

### 🌱 Organic Treatment Library
- **OMRI-Approved Solutions** - Only organic-certified treatments
- **IPM Methodology** - Integrated Pest Management principles
- **Severity-Based Recommendations** - Scaled responses based on infestation level
- **Prevention Strategies** - Long-term management approaches

### 📱 User-Friendly Interface
- **Streamlit Web Application** - Modern, responsive design
- **Mobile-Friendly** - Works on phones, tablets, and desktops
- **Offline-First Design** - Core functionality works without internet
- **Progress Indicators** - Real-time feedback during processing

## 🐛 Supported Pest Types (12 Classes)

### Beneficial Insects
- **Ants** - Usually beneficial soil aerators
- **Bees** - Essential pollinators (identification for protection)
- **Earthworms** - Soil health indicators

### Crop Pests
- **Beetles** - Various species affecting roots and foliage
- **Caterpillars** - Leaf-eating larvae of moths and butterflies
- **Grasshoppers** - Foliage damage and crop destruction
- **Moths** - Adult stage of caterpillars, indicator species
- **Slugs** - Soft-bodied mollusks damaging seedlings
- **Snails** - Shell-bearing mollusks affecting tender plants
- **Weevils** - Specialized beetles targeting specific crops

### Mixed Impact
- **Earwigs** - Can be both beneficial and pest
- **Wasps** - Predatory insects with complex ecosystem roles

## � Project Structure

```
├── 📄 main.py                 # Core system orchestrator
├── 🚀 start.py               # Smart application launcher
├── 📋 requirements.txt        # Python dependencies
├── 📖 README.md              # This file
│
├── �📱 mobile/                # Web interface (Streamlit)
│   ├── app_interface.py      # Main UI components
│   └── __init__.py
│
├── 👁️ vision/                # Computer vision module
│   ├── pest_detector.py      # EfficientNet-B0 implementation
│   └── __init__.py
│
├── 🌱 treatments/            # Treatment recommendation engine
│   ├── recommendation_engine.py  # Organic treatment database
│   └── __init__.py
│
├── 💬 conversation/          # Chat system
│   ├── llm_integration.py    # LM Studio integration
│   ├── chat_interface.py     # Chat UI components
│   └── __init__.py
│
├── 🔧 edge/                  # Edge optimization (future)
│   ├── model_optimizer.py    # ONNX conversion utilities
│   └── __init__.py
│
├── 🧪 tests/                 # Test suite
│   └── test_system.py        # System integration tests
│
├── 📊 models/                # Pre-trained models
│   ├── pest_classifier/      # EfficientNet-B0 weights
│   │   └── weights/
│   │       ├── best.pt       # Best performing model
│   │       ├── last.pt       # Latest checkpoint
│   │       └── epoch*.pt     # Training checkpoints
│   ├── yolov8n-cls.pt       # YOLOv8 classifier (backup)
│   └── optimized/           # Edge-optimized models
│
├── 📂 datasets/             # Training datasets (12 pest classes)
│   ├── ants/
│   ├── bees/
│   ├── beetles/
│   └── ... (9 more classes)
│
├── 🏋️ training/             # Model training scripts
│   ├── improved_train.py     # Main training pipeline
│   ├── evaluate_model.py     # Model evaluation
│   └── datasets_split/       # Train/validation splits
│
├── 📝 logs/                 # Application logs
│   └── pest_management.log
│
├── 🌐 locales/              # Internationalization
│   └── en.json
│
└── 📚 docs/                 # Documentation
    └── Final Project Topic - Organic Farm Pest Management AI System.md
```

## 🛠️ Technical Architecture

### Machine Learning Stack
- **Primary Model**: EfficientNet-B0 with custom classification head
- **Framework**: PyTorch 2.0+ with torchvision
- **Uncertainty Quantification**: Monte Carlo Dropout + Temperature Scaling
- **Fallback Models**: YOLOv8 classification, basic ML classifiers
- **Model Format**: PyTorch (.pt) with ONNX export capability

### Web Application
- **Framework**: Streamlit 1.28+
- **Image Processing**: PIL (Pillow) for preprocessing
- **UI Components**: Custom CSS styling with responsive design
- **Session Management**: Streamlit session state for user experience

### Conversational AI
- **Integration**: LM Studio API client
- **Local Inference**: Supports Llama-2, Code Llama, and compatible models
- **Fallback**: Rule-based responses when LLM unavailable
- **Context**: Pest identification results integrated into conversations

### Dependencies Management
- **Core Dependencies**: Streamlit, Pillow, NumPy (minimal install)
- **Enhanced Features**: PyTorch, torchvision, scikit-learn (full ML stack)
- **Optional Components**: ONNX runtime, OpenCV (edge optimization)
- **Smart Installation**: `start.py` handles dependency checking and installation

## � Usage Guide

### Basic Pest Identification
1. **Launch Application**: Run `python start.py`
2. **Upload Image**: Navigate to "Pest Identification" and upload a clear photo
3. **Get Results**: Receive instant identification with confidence scores
4. **View Treatments**: Access organic treatment recommendations

### Chat Assistant
1. **Navigate to Chat**: Click "Chat Assistant" in the sidebar
2. **Ask Questions**: Type natural language questions about pest management
3. **Context Integration**: Previous identifications automatically inform responses
4. **Treatment Guidance**: Get detailed organic farming advice

### Treatment Library
1. **Browse Treatments**: Access comprehensive organic treatment database
2. **Filter by Pest**: View pest-specific recommendations
3. **IPM Principles**: Learn integrated pest management strategies
4. **Prevention Tips**: Access long-term management advice

## 🧪 Testing

Run the test suite to verify system functionality:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test
python tests/test_system.py
```

Tests cover:
- Pest detection accuracy
- Treatment recommendation engine
- LM Studio integration
- UI component functionality
- Error handling and graceful degradation

## ⚙️ Configuration

### Environment Variables
Create a `.env` file for custom configuration:

```bash
# LM Studio API Configuration
LM_STUDIO_API_URL=http://localhost:1234/v1
LM_STUDIO_MODEL=llama-2-7b-chat

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/pest_management.log

# Model Configuration
MODEL_PATH=models/pest_classifier/weights/best.pt
CONFIDENCE_THRESHOLD=0.7
```

### Customizing Models
- Place custom trained models in `models/pest_classifier/weights/`
- Update model paths in `vision/pest_detector.py`
- Retrain using scripts in `training/` directory

## 🌐 Deployment Options

### Local Development
```bash
python start.py  # Launches on http://localhost:8501
```

### Edge Device Deployment
```bash
# Install with edge optimization
python start.py --enhanced

# Run with limited resources
python start.py --console  # Command-line interface
```

### Cloud Deployment
The application can be deployed on cloud platforms supporting Python and Streamlit:
- **Streamlit Cloud**: Direct deployment from GitHub
- **Heroku**: Use included `Procfile` (if available)
- **Docker**: Container support for consistent deployment

## 🤝 Contributing

1. **Fork the Repository**
2. **Create Feature Branch**: `git checkout -b feature/amazing-feature`
3. **Commit Changes**: `git commit -m 'Add amazing feature'`
4. **Push to Branch**: `git push origin feature/amazing-feature`
5. **Open Pull Request**

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run code formatting
black .

# Run linting
flake8 .

# Run tests
pytest tests/ -v
```

## � Development Team

This project was developed by a collaborative team from **Singapore Institute of Technology (SIT)** as part of their Final Project for the Overseas Immersion Programme:

### **Information and Communications Technology (ICT) Students**
- **Ryan Koo Wei Feng** - Information Security (IS)
- **Farihin Fatten Binte Abdul Rahman** - Information Security (IS)  
- **Khoo Ye Chen** - Software Engineering (SE)
- **Gan Kang Ting, Ryan** - Information Security (IS)
- **Donovan Leong Jia Le** - Applied Artificial Intelligence (AI)

### **Academic Collaboration**
- 🏫 **Singapore Institute of Technology (SIT)**
- 🎓 **FPT University Da Nang** (Overseas Immersion Programme)

### **Project Highlights**
- 🎓 Final Project for Overseas Immersion Programme
- 🚀 Real-world agricultural technology application
- 🤝 Interdisciplinary team approach (Software Engineering + Information Security + AI)
- 🌏 International academic collaboration

## �📄 License

This project is developed for educational purposes as part of an Overseas Immersion Programme. Please respect academic integrity guidelines when using or referencing this code.

## � Acknowledgments

- **EfficientNet**: Efficient neural network architecture for computer vision
- **LM Studio**: Local LLM inference platform
- **Streamlit**: Rapid web application development framework
- **PyTorch**: Deep learning framework and ecosystem
- **Organic Farming Community**: Domain expertise and testing feedback

## 📞 Support

For issues, questions, or contributions:
- **GitHub Issues**: Report bugs and request features
- **Documentation**: Check `docs/` directory for detailed information
- **Logs**: Review `logs/pest_management.log` for troubleshooting

---

**🌱 Supporting organic farmers with AI-powered pest management solutions**
├── 🚀 start.py               # Universal starter with dependency management
├── 📋 requirements.txt       # Project dependencies
│
├── 📖 docs/                  # Documentation
│   ├── PROJECT_BRIEF.md      # Original project requirements
│   ├── FINAL_PROJECT_STATUS.md
│   ├── REQUIREMENTS_COMPLIANCE.md
│   └── MODEL_TRAINING_GUIDE.md
│
├── 👁️ vision/               # Computer vision modules
│   ├── pest_detector.py     # Base detector
│   ├── pest_detector_enhanced.py  # YOLOv8 implementation
│   └── pest_detector_production.py  # Production optimized
│
├── 🌿 treatments/           # Treatment recommendation engine
│   └── recommendation_engine.py
│
├── 💬 conversation/         # Chat interface
│   └── chat_interface.py
│
├── 📱 mobile/              # Mobile-specific functionality
│   ├── app_interface.py
│   └── clipboard_handler.py
│
├── ⚡ edge/                # Edge computing optimizations
│   └── model_optimizer.py
│
├── 🧠 models/              # Trained models and optimizations
│   ├── optimized/          # ONNX and quantized models
│   ├── pest_classifier/    # Base classification models
│   └── pest_classifier2/
│
├── 🎓 training/            # Model training pipeline
│   ├── train_yolo_model.py # Full training script
│   ├── quick_train.py      # Quick training for testing
│   └── datasets_split/     # Training/validation splits
│
├── 📊 datasets/            # Agricultural pest dataset
├── 🌍 locales/            # Internationalization files
├── 🧪 tests/              # Test files
└── 📝 logs/               # Application logs
```

## 🔧 Technical Details

### Computer Vision
- **Model**: YOLOv8-nano optimized for edge deployment
- **Training**: Custom fine-tuned on Agricultural Pests Dataset
- **Inference**: ONNX format for efficient processing
### Conversational AI
- **Primary**: LM Studio integration with local Llama-2-7B-Chat
- **Fallback**: Rule-based pattern matching for offline scenarios
- **Context Aware**: Pest detection results inform chat responses
- **Automatic Treatment**: Auto-queries LLM when "Chat About Treatment" pressed

### Treatment Recommendations
- **Standards**: OMRI-approved organic treatments only
- **Approach**: Integrated Pest Management (IPM) principles
- **Coverage**: 12 pest classes with specific organic controls
- **Knowledge Base**: Scientific names, damage types, affected crops

### System Architecture
- **Backend**: Python-based modular architecture
- **Frontend**: Streamlit responsive web application
- **Models**: PyTorch with automatic CPU/GPU detection
- **Storage**: Local file system with JSON configurations

## 🎮 Usage Examples

### Web Interface
```bash
# Start the web application
python start.py

# Access at http://localhost:8501
# 1. Upload pest images via drag-drop or camera
# 2. Get instant EfficientNet-based identification
# 3. Click "Chat About Treatment" for LM Studio consultation
# 4. Receive organic treatment recommendations
```

### Model Training (EfficientNet)
```bash
# Train the EfficientNet ensemble
cd training
python improved_train.py

# The training process:
# 1. Loads Agricultural Pest Dataset (12 classes)
# 2. Applies stratified K-fold cross-validation (5 folds)
# 3. Trains EfficientNet-B0 with agricultural augmentations
# 4. Saves models with temperature scaling to models/improved/
```

## 🧠 How the CNN is Trained

### Training Pipeline Overview
The EfficientNet-B0 ensemble is trained using a sophisticated pipeline designed for agricultural pest classification:

#### 1. **Data Preparation**
```python
# Dataset Structure: 12 pest classes from Agricultural Pest Dataset
# - ants, bees, beetle, catterpillar, earthworms, earwig
# - grasshopper, moth, slug, snail, wasp, weevil
# Each class contains 100+ high-quality agricultural images
```

#### 2. **Agricultural-Specific Augmentations**
```python
# Custom augmentations for real farm conditions:
- Random lighting variations (dawn/dusk/cloudy conditions)
- Soil texture overlays and dirt spots simulation
- Weather effects (rain droplets, dust particles)
- Camera angle variations (farmer perspective)
- Color jittering for different lighting conditions
```

#### 3. **Model Architecture**
```python
class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=12, dropout_rate=0.3):
        # Pre-trained EfficientNet-B0 backbone (ImageNet weights)
        self.backbone = models.efficientnet_b0(weights='DEFAULT')
        
        # Custom classifier head:
        # Dropout(0.3) → Linear(1280→512) → ReLU → Dropout(0.15) → Linear(512→12)
        
        # Temperature scaling for uncertainty calibration
        self.temperature = nn.Parameter(torch.ones(1))
```

#### 4. **Training Strategy**
```python
# Stratified K-Fold Cross-Validation (5 folds)
config = {
    'num_epochs': 100,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'batch_size': 32,
    'patience': 15,           # Early stopping
    'min_delta': 1e-4        # Minimum improvement threshold
}

# Each fold trains independently:
# - 80% training, 20% validation
# - Best model saved based on validation accuracy
# - Temperature scaling calibrated post-training
```

#### 5. **Uncertainty Quantification**
```python
# Monte Carlo Dropout during inference:
# - Dropout layers kept active during prediction
# - 20 forward passes per image
# - Mean prediction + standard deviation = uncertainty estimate
# - Temperature scaling calibrates confidence scores
```

#### 6. **Training Results**
```
Fold 0: 93.631% validation accuracy
Fold 1: 93.540% validation accuracy  
Fold 2: 93.358% validation accuracy
Fold 3: 92.903% validation accuracy
Fold 4: 92.077% validation accuracy

Average: 93.1% validation accuracy
Ensemble: >95% expected accuracy due to model diversity
```

#### 7. **Production Deployment**
```python
# All 5 models loaded as ensemble:
# - Inference runs on all models simultaneously
# - Predictions averaged for final result
# - Uncertainty calculated across models
# - Real-time performance: ~2-3 seconds per image
```

## 🏆 Project Achievements

- ✅ **EfficientNet-B0 Ensemble** - 93.1% average validation accuracy across 5 folds
- ✅ **12 pest species detection** - Comprehensive agricultural coverage  
- ✅ **LM Studio Integration** - Local LLM for treatment consultation
- ✅ **Uncertainty Quantification** - Monte Carlo Dropout + Temperature Scaling
- ✅ **Production Ready** - Cleaned architecture, error handling, logging
- ✅ **Auto-Treatment Chat** - Automatic LLM queries for treatment advice

## 📚 Documentation

- [Project Brief](docs/PROJECT_BRIEF.md) - Original requirements and scope
- [Project Status](docs/FINAL_PROJECT_STATUS.md) - Current implementation status
- [Requirements Compliance](docs/REQUIREMENTS_COMPLIANCE.md) - How we meet all requirements
- [Training Guide](docs/MODEL_TRAINING_GUIDE.md) - Model training instructions

## 🔄 Development Status

**Status: COMPLETE ✅**

This project successfully implements all required features:
- Computer vision pest detection using fine-tuned YOLOv8
- Conversational AI interface for farmer guidance
- Comprehensive organic treatment recommendations
- Complete offline operation on edge devices
- Mobile-friendly responsive design

---

*Built for organic farmers, by agricultural AI specialists* 🌱

**Agricultural Dataset (Primary - 12 Species):**
- Ants
- Bees (Beneficial Species)
- Beetles
- Caterpillars
- Earthworms (Beneficial Species)
- Earwigs
- Grasshoppers
- Moths
- Slugs
- Snails
- Wasps
- Weevils

**Legacy Support (7 Additional Species):**
- Aphids
- Spider Mites
- Whitefly
- Thrips
- Colorado Potato Beetle
- Cucumber Beetle
- Flea Beetle

**Total: 19 pest/beneficial species with comprehensive treatment database**

## 📁 Project Structure

```
Final-Project/
│
├── main.py                    # Main application entry point
├── start.py                   # Alternative startup script
├── requirements.txt           # Python dependencies
│
├── vision/                    # Computer vision module
│   ├── __init__.py
│   └── pest_detector.py       # Pest detection with PyTorch
│
├── treatments/                # Treatment recommendation engine
│   ├── __init__.py
│   └── recommendation_engine.py # Organic treatment database and IPM logic
│
├── conversation/              # Conversational AI interface
│   ├── __init__.py
│   └── chat_interface.py      # Natural language processing and chat
│
├── edge/                      # Edge computing optimization
│   ├── __init__.py
│   └── model_optimizer.py     # Model compression and optimization
│
├── mobile/                    # Web/mobile interface
│   ├── __init__.py
│   └── app_interface.py       # Streamlit web application
│
├── tests/                     # Test suite
│   └── test_system.py         # Unit and integration tests
│
├── locales/                   # Internationalization
│   └── en.json                # English UI strings
│
├── datasets/                  # Agricultural Pests Image Dataset (from Kaggle)
│   ├── ants/                  # Ant images
│   ├── bees/                  # Bee images (beneficial)
│   ├── beetle/                # Beetle images
│   ├── catterpillar/          # Caterpillar images
│   ├── earthworms/            # Earthworm images (beneficial)
│   ├── earwig/                # Earwig images
│   ├── grasshopper/           # Grasshopper images
│   ├── moth/                  # Moth images
│   ├── slug/                  # Slug images
│   ├── snail/                 # Snail images
│   ├── wasp/                  # Wasp images
│   └── weevil/                # Weevil images
│
├── models/                    # AI models (created at runtime)
│   └── optimized/             # Edge-optimized models
│
└── Final Project Topic - Organic Farm Pest Management AI System.md
```

## 🌐 How to Use

### 1. Pest Identification

1. **Launch the application** using `streamlit run start.py`
2. **Navigate to "Pest Identification"** in the sidebar
3. **Try demo images** from the agricultural dataset, or
4. **Upload your own image** (PNG, JPG, JPEG supported)
5. **Click "Analyze"** to get instant identification

**Results include:**
   - Pest species name (scientific and common)
   - Confidence score
   - Severity assessment (Low/Medium/High)
   - Affected crops information
   - Beneficial species recognition

### 2. Treatment Recommendations

1. **Automatic recommendations** based on identification
2. **Severity-based treatment plans**:
   - **Low**: Cultural controls and monitoring
   - **Medium**: Biological and mechanical controls
   - **High**: Comprehensive IPM approach
3. **Organic compliance** verification for all treatments

### 3. Chat Assistant

1. **Ask questions** in natural language
2. **Get context-aware responses** based on your pest situation
3. **Request specific information** about:
   - Treatment application timing
   - Organic certification requirements
   - Cost-effective solutions
   - Prevention strategies

## 📊 Agricultural Pests Image Dataset Integration

This system uses the comprehensive [Agricultural Pests Image Dataset](https://www.kaggle.com/datasets/vencerlanz09/agricultural-pests-image-dataset) from Kaggle, featuring 12 different types of agricultural pests with 300px maximum resolution images.

### Dataset Features
- **12 Pest Types**: Complete coverage of common agricultural pests
- **Beneficial Species**: Automatically identifies bees and earthworms as beneficial organisms
- **Treatment Mapping**: Each pest type maps to comprehensive organic treatment protocols
- **Demo Integration**: Web interface includes sample images from the dataset for testing

### Usage Requirements
1. **Download** the Agricultural Pests Image Dataset from Kaggle
2. **Extract** to `datasets/` directory in the project root
3. **Verify** the structure matches the project layout above
4. **Run** the system - photo examples will be automatically available

## 🧪 Testing

Run the comprehensive test suite to verify system functionality:

```bash
# Core system tests
python tests/test_system.py
```

### Test Coverage
The test suite validates:
- ✅ Computer vision pest detection (19 species)
- ✅ Treatment recommendation engine (17 pest categories)
- ✅ Conversational AI interface
- ✅ Edge optimization
- ✅ System integration
- ✅ Agricultural dataset compatibility
- ✅ Beneficial species detection
- ✅ Data integrity and organic compliance

## 🔧 Technical Specifications

### System Requirements
- **Python**: 3.8 or higher
- **Memory**: 4GB RAM minimum (8GB recommended)
- **Storage**: 2GB for system + dataset storage
- **Network**: Optional (system works fully offline)

### Performance Targets
- **Processing Speed**: <200ms inference time
- **Accuracy**: High confidence scoring with probabilistic outputs
- **Organic Compliance**: 100% OMRI-approved treatments
- **Edge Computing**: ONNX optimization ready
- **Offline Operation**: Complete functionality without internet

### Dependencies
Key packages include:
- `streamlit` - Web interface
- `Pillow` - Image processing
- `torch` - Deep learning (optional, graceful degradation)
- `pathlib` - File system operations

## 🌐 Offline Operation

The system is designed for complete offline operation:

- **No Internet Required**: All core functionality works without connectivity
- **Edge Computing**: Optimized for resource-constrained devices
- **Graceful Degradation**: Falls back to demo mode if ML dependencies unavailable
- **Local Storage**: All data and models stored locally

## 🎯 Project Requirements Compliance

This system fully satisfies all project requirements:

✅ **Intelligent, conversational AI system** - Complete chat interface
✅ **Advanced computer vision capabilities** - 19 pest species support  
✅ **Comprehensive pest management consultant** - Expert recommendations
✅ **Offline operation on edge devices** - Full offline functionality
✅ **Image capture/upload functionality** - Streamlit interface + demo images
✅ **Immediate, expert-level identification** - <200ms processing
✅ **Tailored organic treatment recommendations** - OMRI-approved only
✅ **Use of developed/fine-tuned models** - Custom architecture ready

## 🚀 Getting Started

1. **Download the Agricultural Pests Image Dataset** from Kaggle
2. **Extract to `datasets/`** in the project directory
3. **Install dependencies**: `pip install -r requirements.txt`
4. **Run the system**: `streamlit run start.py`
5. **Open your browser** to the displayed URL (typically http://localhost:8501)
6. **See photography tips** with good example photos from each pest category
7. **Upload your own pest photos** and get instant identification and organic treatment recommendations

## 📞 System Commands

When running in console mode (`python main.py`):
- `test` - Run system test with sample image
- `quit` - Exit the application
- `help` - Show available commands

## 🎉 Ready to Use!

Your Organic Farm Pest Management AI System is ready to help farmers identify pests and get organic treatment recommendations. The system combines the power of computer vision with comprehensive agricultural knowledge to provide instant, expert-level guidance for sustainable farming practices.

**🌱 Supporting organic farming through AI technology! 🌱**
