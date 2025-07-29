"""
Setup script for Organic Farm Pest Management AI System
Handles initial setup, dependency verification, and system configuration.
"""

import subprocess
import sys
import os
from pathlib import Path
import logging

def setup_logging():
    """Configure logging for setup process."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - SETUP - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible."""
    logger = logging.getLogger(__name__)
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error(f"Python 3.8+ required, but found {version.major}.{version.minor}")
        return False
    
    logger.info(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_dependencies():
    """Install required Python packages."""
    logger = logging.getLogger(__name__)
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        logger.error("❌ requirements.txt not found")
        return False
    
    try:
        logger.info("📦 Installing dependencies...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            logger.info("✅ Dependencies installed successfully")
            return True
        else:
            logger.error(f"❌ Failed to install dependencies: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Installation timed out")
        return False
    except Exception as e:
        logger.error(f"❌ Installation error: {str(e)}")
        return False

def create_directories():
    """Create necessary directories."""
    logger = logging.getLogger(__name__)
    
    directories = [
        "models",
        "models/optimized",
        "data",
        "logs"
    ]
    
    project_root = Path(__file__).parent
    
    for directory in directories:
        dir_path = project_root / directory
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ Created directory: {directory}")
        except Exception as e:
            logger.error(f"❌ Failed to create directory {directory}: {str(e)}")
            return False
    
    return True

def verify_imports():
    """Verify that critical imports work."""
    logger = logging.getLogger(__name__)
    
    critical_imports = [
        ("PIL", "Pillow for image processing"),
        ("streamlit", "Streamlit for web interface"),
        ("numpy", "NumPy for numerical operations")
    ]
    
    optional_imports = [
        ("torch", "PyTorch for deep learning"),
        ("cv2", "OpenCV for computer vision"),
        ("onnx", "ONNX for model optimization")
    ]
    
    # Check critical imports
    for module, description in critical_imports:
        try:
            __import__(module)
            logger.info(f"✅ {description}")
        except ImportError:
            logger.error(f"❌ Missing critical dependency: {module} ({description})")
            return False
    
    # Check optional imports
    for module, description in optional_imports:
        try:
            __import__(module)
            logger.info(f"✅ {description}")
        except ImportError:
            logger.warning(f"⚠️ Optional dependency missing: {module} ({description})")
    
    return True

def test_system_basic():
    """Run basic system tests."""
    logger = logging.getLogger(__name__)
    
    try:
        # Test basic imports
        from main import PestManagementSystem
        logger.info("✅ Main system imports successfully")
        
        # Test system initialization
        system = PestManagementSystem()
        logger.info("✅ System initializes successfully")
        
        # Test chat interface
        response = system.chat_with_system("Hello")
        if response and len(response) > 0:
            logger.info("✅ Chat interface responding")
        else:
            logger.warning("⚠️ Chat interface may have issues")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Basic system test failed: {str(e)}")
        return False

def create_startup_script():
    """Create platform-specific startup scripts."""
    logger = logging.getLogger(__name__)
    
    project_root = Path(__file__).parent
    
    # Windows batch script
    batch_content = """@echo off
echo Starting Organic Farm Pest Management AI System...
echo.
cd /d "%~dp0"
python main.py
pause
"""
    
    batch_file = project_root / "start_system.bat"
    try:
        with open(batch_file, 'w') as f:
            f.write(batch_content)
        logger.info("✅ Created Windows startup script: start_system.bat")
    except Exception as e:
        logger.warning(f"⚠️ Could not create Windows script: {str(e)}")
    
    # Unix shell script
    shell_content = """#!/bin/bash
echo "Starting Organic Farm Pest Management AI System..."
echo
cd "$(dirname "$0")"
python3 main.py
"""
    
    shell_file = project_root / "start_system.sh"
    try:
        with open(shell_file, 'w') as f:
            f.write(shell_content)
        
        # Make executable on Unix systems
        if os.name != 'nt':
            os.chmod(shell_file, 0o755)
        
        logger.info("✅ Created Unix startup script: start_system.sh")
    except Exception as e:
        logger.warning(f"⚠️ Could not create Unix script: {str(e)}")

def show_setup_summary():
    """Display setup completion summary."""
    logger = logging.getLogger(__name__)
    
    print("\n" + "="*60)
    print("🌱 ORGANIC FARM PEST MANAGEMENT AI SYSTEM")
    print("="*60)
    print("✅ Setup completed successfully!")
    print()
    print("🚀 TO START THE SYSTEM:")
    print("   Option 1: Run 'python main.py'")
    print("   Option 2: Double-click 'start_system.bat' (Windows)")
    print("   Option 3: Run './start_system.sh' (Unix/Linux/Mac)")
    print()
    print("🌐 WEB INTERFACE:")
    print("   http://localhost:8501")
    print()
    print("📚 FEATURES:")
    print("   • 🔍 Pest identification from photos")
    print("   • 💬 AI chat assistant")
    print("   • 🌱 Organic treatment recommendations")
    print("   • 📱 Mobile-friendly interface")
    print("   • ⚡ Offline operation")
    print()
    print("🆘 TROUBLESHOOTING:")
    print("   • Check README.md for detailed instructions")
    print("   • Run 'python tests/test_system.py' to verify installation")
    print("   • Use the built-in chat assistant for help")
    print("="*60)
    print("🌾 Happy Organic Farming! 🌾")
    print("="*60)

def main():
    """Main setup process."""
    logger = setup_logging()
    
    print("🌱 Setting up Organic Farm Pest Management AI System...")
    print("="*60)
    
    # Setup steps
    steps = [
        ("Checking Python version", check_python_version),
        ("Installing dependencies", install_dependencies),
        ("Creating directories", create_directories),
        ("Verifying imports", verify_imports),
        ("Testing basic system", test_system_basic),
        ("Creating startup scripts", create_startup_script)
    ]
    
    for step_name, step_function in steps:
        logger.info(f"🔄 {step_name}...")
        try:
            success = step_function()
            if not success:
                logger.error(f"❌ Setup failed at: {step_name}")
                return False
        except Exception as e:
            logger.error(f"❌ Setup failed at {step_name}: {str(e)}")
            return False
    
    # Show summary
    show_setup_summary()
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Setup failed. Please check the error messages above.")
        sys.exit(1)
    else:
        print("\n✅ Setup completed successfully!")
        sys.exit(0)
