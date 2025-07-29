# Organic Farm Pest Management AI System - Copilot Instructions

## Project Overview
This is a demo-capable, offline-first AI system for organic farmers. The system uses a modular architecture with graceful degradation - heavy ML dependencies are optional, falling back to demo implementations when unavailable.

## Architecture & Key Components

### System Entry Point
- **`main.py`**: Central orchestrator using `PestManagementSystem` class
- **Graceful Initialization**: All modules imported with try/catch, system continues if components fail
- **Streamlit Web Interface**: Launch via `python main.py` → accessible at `http://localhost:8501`

### Module Architecture (Dependency Injection Pattern)
```python
# main.py pattern - modules are optional
try:
    from vision.pest_detector import PestDetector
    self.pest_detector = PestDetector()
except ImportError:
    self.pest_detector = None
```

### Core Modules
- **`vision/`**: Computer vision with demo fallback (`pest_detector.py` → `pest_detector_demo.py`)
- **`treatments/`**: Organic treatment database and IPM logic (`recommendation_engine.py`)
- **`conversation/`**: Template-based chat interface with farmer context (`chat_interface.py`)
- **`mobile/`**: Streamlit web app with custom CSS and farmer-friendly UI (`app_interface.py`)
- **`edge/`**: ONNX model optimization and edge deployment tools (`model_optimizer.py`)

## Critical Development Patterns

### Dependency Management
- **Optional Heavy Dependencies**: ML libraries (torch, cv2) are commented out in `requirements.txt`
- **Demo Mode**: System works without ML dependencies using simulation
- **Production Setup**: Uncomment ML dependencies for full computer vision

### Error Handling & Logging
```python
# Standard pattern across all modules
import logging
logger = logging.getLogger(__name__)
logger.info("Module initialized successfully")
```
- **Central Logging**: All output goes to `pest_management.log`
- **Graceful Degradation**: Missing modules don't crash the system

### Data Structures
- **Pest Classes**: 8 supported pests with scientific names, severity indicators, affected crops
- **Treatment Categories**: biological, cultural, mechanical (all organic-certified)
- **IPM Principles**: Integrated in treatment recommendations with severity-based escalation

## Essential Workflows

### Quick Start
```powershell
pip install -r requirements.txt  # Core dependencies only
python main.py                   # Launches Streamlit interface
# Browser opens to http://localhost:8501
```

### Testing
```powershell
cd tests
python test_system.py           # Comprehensive test suite
```

### Demo Content
- **Test Images**: `test_images/` contains pest images with quality levels (high/medium/low)
- **Pest Simulation**: Demo detector analyzes filenames to simulate identification

## Module-Specific Patterns

### Vision Module (`vision/pest_detector.py`)
```python
# Fallback pattern for optional dependencies
try:
    import cv2, torch  # Heavy ML dependencies
    # Full implementation here
except ImportError:
    from .pest_detector_demo import PestDetector  # Demo fallback
```

### Treatment Engine (`treatments/recommendation_engine.py`)
- **Database Structure**: Nested dict with pest → treatment_type → methods
- **Severity-Based Recommendations**: Different strategies for low/medium/high infestations
- **Organic Compliance**: All treatments marked `organic_certified: True`

### Chat Interface (`conversation/chat_interface.py`)
- **Template-Based Responses**: Predefined response categories (greeting, treatment_explanation, etc.)
- **Context Awareness**: Maintains conversation history and pest identification context
- **Farmer-Friendly Language**: Conversational tone, encouragement, practical advice

### Streamlit App (`mobile/app_interface.py`)
- **Custom CSS Styling**: Organic green theme, styled boxes for results
- **Image Upload**: PIL-based image handling with file upload widget
- **Multi-Tab Interface**: Separate tabs for identification, treatments, chat, testing

## Edge Deployment Considerations

### Model Optimization (`edge/model_optimizer.py`)
- **ONNX Conversion**: Converts PyTorch models to ONNX for edge deployment
- **Size Targets**: Pest detection <50MB, treatment engine <10MB
- **Performance Monitoring**: Tracks model size, inference time, accuracy

### Resource Management
- **Efficient Storage**: Models saved to `models/optimized/`
- **Memory-Conscious**: Designed for 4GB+ RAM environments
- **Battery Optimization**: Edge-optimized inference for mobile deployment

When extending this system, maintain the graceful degradation pattern and ensure all new features work in demo mode.

When working on this project, prioritize offline functionality and farmer usability over technical complexity. Every feature should work reliably in remote farm locations with poor connectivity.
