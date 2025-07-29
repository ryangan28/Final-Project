# 🌱 Organic Farm Pest Management AI System

An intelligent, offline-first pest management system designed for organic farmers. This system combines computer vision, conversational AI, and edge computing to provide real-time pest identification and organic treatment recommendations.

## 🚀 Quick Start

### Demo Mode (Recommended for First-Time Users)
```bash
# Clone repository
git clone https://github.com/ryangan28/Final-Project.git
cd Final-Project

# Install lightweight dependencies
pip install -r requirements-demo.txt

# Run test suite (should show 100% pass rate)
python tests/test_system.py

# Start demo application
streamlit run mobile/app_interface.py
```

### Production Mode (Full ML Capabilities)
```bash
# Install complete dependencies
pip install -r requirements-full.txt

# Run full system
streamlit run mobile/app_interface.py
```

> 📖 **Detailed Instructions**: See [`docs/installation.md`](docs/installation.md) for complete setup guide

## 🎯 Features

### ✅ Core Capabilities
- **🔍 Computer Vision Pest Detection**: Identify 8 common agricultural pests from photos
- **💬 Conversational AI Assistant**: Natural language interaction for guidance
- **🌱 Organic Treatment Recommendations**: OMRI-approved treatments only
- **📱 Mobile-Friendly Interface**: Works on desktop and mobile devices
- **⚡ Edge Computing Optimized**: Runs offline on resource-constrained devices
- **🔄 Integrated Pest Management**: IPM-based approach for sustainable control

### 🐛 Supported Pests
- Aphids
- Caterpillars (Lepidoptera larvae)
- Spider Mites
- Whitefly
- Thrips
- Colorado Potato Beetle
- Cucumber Beetle
- Flea Beetle

### 🌾 Treatment Categories
- **Biological Controls**: Beneficial insects, microbial pesticides
- **Cultural Controls**: Crop rotation, companion planting, habitat modification
- **Mechanical Controls**: Physical barriers, traps, manual removal
- **Preventive Measures**: IPM strategies for long-term pest management

## 📊 Performance Metrics
- **Accuracy**: 87%+ (verified in [`docs/evaluation.md`](docs/evaluation.md))
- **Inference Speed**: <200ms on edge devices
- **Model Size**: <50MB (edge-optimized)
- **Offline Capability**: 100% functionality without internet
- **Test Coverage**: 100% (27/27 tests passing)

## 📁 Project Structure

```
Final-Project/
├── main.py                     # Main application entry point
├── requirements-demo.txt       # Lightweight dependencies
├── requirements-full.txt       # Complete ML dependencies  
├── requirements.txt            # Legacy requirements file
├── README.md                  # This file
├── CHANGELOG.md               # Version history and changes
├── .gitignore                 # Git ignore patterns
│
├── vision/                    # Computer vision module
│   ├── __init__.py
│   ├── pest_detector.py       # Full pest detection with PyTorch
│   └── pest_detector_demo.py  # Demo version with fallbacks
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
├── tests/                     # Comprehensive test suite
│   └── test_system.py         # Unit and integration tests
│
├── docs/                      # Documentation
│   ├── installation.md        # Setup guide
│   └── evaluation.md          # Performance metrics
│
├── locales/                   # Internationalization
│   └── en.json                # English UI strings
│
├── test_images/               # Sample images for testing
│   ├── aphids_high.jpg
│   ├── aphids_medium.jpg
│   ├── aphids_low.jpg
│   └── [other pest images...]
│
├── models/                    # AI models (created at runtime)
│   └── optimized/             # Edge-optimized models
│
└── data/                      # Data storage
```

## 🖥️ System Requirements

### Minimum Requirements
- **OS**: Windows 10, macOS 10.14, Ubuntu 18.04, or newer
- **Python**: 3.8 or newer
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space
- **Internet**: Required for initial setup only

### Recommended for Edge Deployment
- **RAM**: 8GB or more
- **CPU**: Multi-core processor
- **GPU**: Optional but recommended for faster inference
- **Storage**: SSD for better performance

## 🚀 Usage Guide

### 1. Pest Identification

1. **Take a clear photo** of the pest or damage
2. **Upload the image** using the web interface
3. **Review the results** including:
   - Pest type identification
   - Confidence level
   - Severity assessment
   - Affected crops information

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

### 4. Treatment Library

1. **Browse treatments** by category or pest type
2. **View detailed information** including:
   - Application instructions
   - Effectiveness ratings
   - Cost estimates
   - Organic certification status

## 🧪 Testing

Run the comprehensive test suite to verify system functionality:

```bash
cd tests
python test_system.py
```

The test suite covers:
- ✅ Computer vision pest detection
- ✅ Treatment recommendation engine
- ✅ Conversational AI interface
- ✅ Edge optimization
- ✅ System integration
- ✅ Data integrity and organic compliance

## 🔧 Configuration

### Model Optimization
The system automatically optimizes models for edge deployment. Configuration options in `edge/model_optimizer.py`:

```python
optimization_configs = {
    'pest_detection': {
        'target_size_mb': 50,      # Maximum model size
        'min_accuracy': 0.85,      # Minimum accuracy threshold
        'quantization': True,      # Enable quantization
        'pruning': True           # Enable model pruning
    }
}
```

### Logging
Logging configuration in `main.py`:
- **Log Level**: INFO (configurable)
- **Log File**: `pest_management.log`
- **Console Output**: Enabled

## 🌐 Offline Operation

The system is designed for offline-first operation:

1. **Model Storage**: All AI models are stored locally
2. **Treatment Database**: Complete organic treatment database is embedded
3. **No Internet Required**: Full functionality without network connectivity
4. **Edge Optimization**: Models optimized for resource-constrained devices

## 🏆 Organic Certification Compliance

All treatment recommendations comply with organic farming standards:

- ✅ **OMRI-Approved**: Only OMRI-listed materials recommended
- ✅ **No Synthetic Pesticides**: Biological and natural controls only
- ✅ **IPM Principles**: Integrated approach emphasizing prevention
- ✅ **Certification Safe**: Maintains organic certification status

## 📊 Performance Metrics

### Model Performance
- **Accuracy**: 87%+ on pest identification
- **Inference Time**: <200ms per image
- **Model Size**: <50MB optimized
- **Confidence Threshold**: 70% for recommendations

### System Performance
- **Startup Time**: <10 seconds
- **Response Time**: <1 second for chat interactions
- **Memory Usage**: <2GB typical operation
- **Offline Capability**: 100% functionality

## � Documentation

### Getting Started
- [`docs/installation.md`](docs/installation.md) - Complete installation guide for demo and production modes
- [`docs/evaluation.md`](docs/evaluation.md) - Detailed performance metrics and validation methodology
- [`locales/en.json`](locales/en.json) - UI text strings for internationalization

### Key Documents
- **Installation Guide**: Step-by-step setup for both demo and production environments
- **Evaluation Report**: 87%+ accuracy validation with field trial results
- **System Architecture**: Modular design supporting offline-first operation
- **Organic Compliance**: OMRI certification and IPM principle adherence

### API Documentation
The system provides modular APIs for integration:

```python
# Pest Detection
from vision.pest_detector import PestDetector
detector = PestDetector()
results = detector.detect('path/to/image.jpg')

# Treatment Recommendations  
from treatments.recommendation_engine import TreatmentEngine
engine = TreatmentEngine()
treatments = engine.get_treatments('Aphids', 'medium')

# Chat Interface
from conversation.chat_interface import ChatInterface
chat = ChatInterface()
response = chat.process_message('How do I treat aphids?')
```

## 🛠️ Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version compatibility (3.8+)

2. **Model Loading Issues**
   - Models are created automatically on first run
   - Check available disk space (2GB required)

3. **Performance Issues**
   - Increase available RAM
   - Consider GPU acceleration if available
   - Check system status page in the web interface

4. **Web Interface Issues**
   - Ensure port 8501 is not in use
   - Try accessing `http://127.0.0.1:8501` instead
   - Check firewall settings

### Getting Help

1. **System Status Page**: Check the built-in diagnostics
2. **Log Files**: Review `pest_management.log` for error details
3. **Test Suite**: Run tests to identify specific issues
4. **Chat Assistant**: Ask the AI for troubleshooting help

## 🔮 Future Enhancements

Potential improvements for future versions:

- **Expanded Pest Database**: Additional pest species
- **Crop-Specific Recommendations**: Tailored by crop type
- **Weather Integration**: Weather-based treatment timing
- **Multi-Language Support**: International accessibility
- **Mobile App**: Native mobile applications
- **IoT Integration**: Smart trap and sensor connectivity

## 📄 License

This project is developed for academic and educational purposes. Please ensure compliance with organic certification requirements in your specific region.

## 🙏 Acknowledgments

- **Organic Farming Research**: Based on established IPM principles
- **Agricultural Extension Services**: Treatment recommendations sourced from expert guidance
- **Open Source Libraries**: Built on PyTorch, Streamlit, and other open-source tools
- **Organic Materials Review Institute (OMRI)**: Treatment compliance verification

---

**🌱 Happy Organic Farming! 🌱**

For questions or support, use the built-in chat assistant or consult your local agricultural extension service.
