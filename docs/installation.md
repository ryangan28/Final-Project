# Installation Guide

## Quick Start

### Option 1: Demo Mode (Lightweight)
For testing and demonstrations without heavy ML dependencies:

```bash
# Clone repository
git clone https://github.com/ryangan28/Final-Project.git
cd Final-Project

# Install demo dependencies
pip install -r requirements-demo.txt

# Run tests
python -m pytest tests/

# Start demo application
streamlit run mobile/app_interface.py
```

### Option 2: Full Production Mode
For production deployment with complete ML capabilities:

```bash
# Clone repository
git clone https://github.com/ryangan28/Final-Project.git
cd Final-Project

# Install full dependencies
pip install -r requirements-full.txt

# Run full test suite
python tests/test_system.py

# Start production application
streamlit run mobile/app_interface.py
```

## Dependency Modes

### Demo Mode Features
- ✅ Pest identification simulation
- ✅ Treatment recommendations
- ✅ Chat interface
- ✅ Web interface
- ✅ All tests passing
- ❌ Real computer vision
- ❌ Edge optimization
- ❌ Model training

### Full Mode Features
- ✅ All demo mode features
- ✅ Real PyTorch computer vision
- ✅ ONNX edge optimization
- ✅ System performance monitoring
- ✅ Model training capabilities
- ✅ Advanced analytics

## System Requirements

### Minimum (Demo Mode)
- **OS**: Windows 10+, macOS 10.14+, Ubuntu 18.04+
- **Python**: 3.8 or higher
- **RAM**: 2GB available
- **Storage**: 100MB

### Recommended (Full Mode)
- **OS**: Windows 11, macOS 12+, Ubuntu 20.04+
- **Python**: 3.9 or higher
- **RAM**: 8GB available (16GB for training)
- **Storage**: 2GB (models and datasets)
- **GPU**: Optional, CUDA-compatible for training

## Environment Setup

### Using Virtual Environment
```bash
# Create virtual environment
python -m venv organic-pest-ai
source organic-pest-ai/bin/activate  # Linux/Mac
# organic-pest-ai\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements-demo.txt  # or requirements-full.txt
```

### Using Conda
```bash
# Create conda environment
conda create -n organic-pest-ai python=3.9
conda activate organic-pest-ai

# Install dependencies
pip install -r requirements-demo.txt  # or requirements-full.txt
```

## Troubleshooting

### Common Issues

#### "Module not found" errors
```bash
# Ensure virtual environment is activated
source organic-pest-ai/bin/activate

# Reinstall requirements
pip install -r requirements-demo.txt --force-reinstall
```

#### PyTorch installation issues
```bash
# For CPU-only PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For CUDA PyTorch (if you have compatible GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### ONNX installation issues
```bash
# Install ONNX runtime
pip install onnxruntime

# For GPU acceleration
pip install onnxruntime-gpu
```

### Dependency Conflicts
If you encounter dependency conflicts:

```bash
# Clean install
pip uninstall -r requirements-full.txt -y
pip install -r requirements-full.txt
```

### Running in Docker

```dockerfile
# Demo mode Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements-demo.txt .
RUN pip install -r requirements-demo.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "mobile/app_interface.py"]
```

```bash
# Build and run
docker build -t organic-pest-ai-demo .
docker run -p 8501:8501 organic-pest-ai-demo
```

## Verification

### Test Installation
```bash
# Run test suite
python tests/test_system.py

# Expected output: 100% tests passed
```

### Check System Status
1. Start the application: `streamlit run mobile/app_interface.py`
2. Navigate to "System Status" page
3. Verify all components show ✅ or ⚠️ with explanations
4. Run component tests to verify functionality

## Development Setup

### For Contributors
```bash
# Install development dependencies
pip install -r requirements-full.txt
pip install pytest black flake8 mypy

# Run linting
black . --check
flake8 .
mypy .

# Run tests
pytest tests/ -v
```

### Pre-commit Hooks
```bash
# Install pre-commit
pip install pre-commit
pre-commit install

# Manual run
pre-commit run --all-files
```

## Performance Optimization

### For Edge Devices
1. Use demo mode for initial testing
2. Install minimal PyTorch for basic CV: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
3. Skip ONNX if not needed for production
4. Monitor memory usage in System Status page

### For Production Servers
1. Install full mode with GPU support
2. Use ONNX optimization for faster inference
3. Enable batch processing for multiple images
4. Monitor performance metrics in web interface

## Support

For installation issues:
1. Check this guide first
2. Review error messages in System Status page
3. Ensure Python version compatibility
4. Create issue on GitHub with system details

---

**Last Updated**: July 29, 2025  
**Compatibility**: Python 3.8+, Windows/macOS/Linux
