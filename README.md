# 🌱 Organic Farm Pest Management AI System

An intelligent, offline-first pest management system designed for organic farmers. This system combines computer vision, conversational AI, and edge computing to provide real-time pest identification and organic treatment recommendations using the Agricultural Pests Image Dataset from Kaggle.

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.8+ required
python --version

# Clone or download this project
# Ensure you have the Agricultural Pests Image Dataset in datasets/
```

### Installation & Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run the system
streamlit run start.py
```

### Alternative Startup
```bash
# Direct main module execution
python main.py

# Or use the start script
python start.py
```

## 🎯 Features

### ✅ Core Capabilities
- **🔍 Computer Vision Pest Detection**: Identify 19 agricultural pest species from photos
- **💬 Conversational AI Assistant**: Natural language interaction for guidance
- **🌱 Organic Treatment Recommendations**: OMRI-approved treatments only
- **📱 Mobile-Friendly Interface**: Works on desktop and mobile devices
- **⚡ Edge Computing Optimized**: Runs offline on resource-constrained devices
- **🔄 Integrated Pest Management**: IPM-based approach for sustainable control

### 🐛 Supported Pests

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
