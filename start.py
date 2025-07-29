#!/usr/bin/env python3
"""
Organic Farm Pest Management AI System - Universal Starter
Smart entry point that handles dependency checking, environment setup, and application launch.

Usage:
    python start.py              # Launch web interface (default)
    python start.py --console    # Launch console interface
    python start.py --setup      # Run setup only
    python start.py --check      # Check dependencies only
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path
import logging
import platform

def setup_logging():
    """Configure logging for startup process."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - STARTUP - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pest_management.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def check_python_version():
    """Verify Python version compatibility."""
    logger = logging.getLogger(__name__)
    
    if sys.version_info < (3, 8):
        logger.error("[ERROR] Python 3.8 or higher is required")
        logger.error(f"Current version: {sys.version}")
        return False
    
    logger.info(f"[OK] Python version check passed: {sys.version.split()[0]}")
    return True

def get_pip_command():
    """Get the appropriate pip command for the platform."""
    commands = ['pip3', 'pip', 'python -m pip', 'python3 -m pip']
    
    for cmd in commands:
        try:
            result = subprocess.run(
                cmd.split() + ['--version'], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if result.returncode == 0:
                return cmd.split()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue
    
    return None

def check_dependencies():
    """Check if required dependencies are installed."""
    logger = logging.getLogger(__name__)
    
    # Core dependencies that must be present
    # Map package names to import names
    required_packages = {
        'streamlit': 'streamlit',
        'Pillow': 'PIL',  # Pillow imports as PIL
        'numpy': 'numpy'
    }
    
    # Optional enhanced packages for better performance
    optional_packages = {
        'onnx': 'onnx',
        'psutil': 'psutil'
    }
    
    missing_packages = []
    optional_missing = []
    
    # Check required packages
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
            logger.info(f"[OK] {package_name} is installed")
        except ImportError:
            missing_packages.append(package_name)
            logger.warning(f"[MISSING] {package_name} is missing")
    
    # Check optional packages
    for package_name, import_name in optional_packages.items():
        try:
            __import__(import_name)
            logger.info(f"[OK] {package_name} is installed (optional)")
        except ImportError:
            optional_missing.append(package_name)
            logger.info(f"[OPTIONAL] {package_name} not installed (enhances performance)")
    
    return missing_packages, optional_missing

def install_dependencies(missing_packages=None, install_optional=False):
    """Install missing dependencies."""
    logger = logging.getLogger(__name__)
    
    if missing_packages is None:
        missing_packages, optional_missing = check_dependencies()
        
        if install_optional:
            missing_packages.extend(optional_missing)
    
    if not missing_packages:
        logger.info("[OK] All dependencies are already installed")
        return True
    
    pip_cmd = get_pip_command()
    if not pip_cmd:
        logger.error("[ERROR] pip not found. Please install pip manually.")
        return False
    
    # Use the single unified requirements file
    requirements_file = "requirements.txt"
    logger.info(f"[INSTALL] Installing dependencies from {requirements_file}...")
    
    try:
        result = subprocess.run(
            pip_cmd + ['install', '-r', requirements_file],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout
        )
        
        if result.returncode == 0:
            logger.info("[OK] Dependencies installed successfully")
            
            # Try installing ONNX specifically for enhanced optimization
            if install_optional or 'onnx' in missing_packages:
                logger.info("[INSTALL] Installing ONNX for enhanced edge optimization...")
                onnx_result = subprocess.run(
                    pip_cmd + ['install', 'onnx>=1.14.0'],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                if onnx_result.returncode == 0:
                    logger.info("[OK] ONNX installed successfully - Enhanced optimization available")
                else:
                    logger.warning("[WARN] ONNX installation failed, continuing with lightweight mode")
            
            return True
        else:
            logger.error(f"[ERROR] Failed to install dependencies: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("[ERROR] Installation timed out")
        return False
    except Exception as e:
        logger.error(f"[ERROR] Installation failed: {str(e)}")
        return False

def check_system_requirements():
    """Check system requirements and configuration."""
    logger = logging.getLogger(__name__)
    
    # Check if we're in the right directory
    if not Path("main.py").exists():
        logger.error("[ERROR] main.py not found. Please run from the project root directory.")
        return False
    
    # Check if required directories exist
    required_dirs = ["vision", "treatments", "conversation", "mobile"]
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            logger.error(f"[ERROR] Required directory '{dir_name}' not found")
            return False
    
    logger.info("[OK] System requirements check passed")
    return True

def launch_web_interface():
    """Launch the Streamlit web interface."""
    logger = logging.getLogger(__name__)
    
    logger.info("[START] Starting web interface...")
    logger.info("[INFO] Access the system at: http://localhost:8501")
    logger.info("[INFO] Upload pest images for identification and treatment recommendations")
    
    try:
        # Use streamlit run with main.py
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'main.py',
            '--server.headless', 'true',
            '--server.port', '8501'
        ])
    except KeyboardInterrupt:
        logger.info("[STOP] Application stopped by user")
    except Exception as e:
        logger.error(f"[ERROR] Failed to start web interface: {str(e)}")
        return False
    
    return True

def launch_console_interface():
    """Launch the console interface."""
    logger = logging.getLogger(__name__)
    
    logger.info("[START] Starting console interface...")
    
    try:
        # Import and run the main system
        from main import PestManagementSystem
        
        system = PestManagementSystem()
        logger.info("[OK] System initialized successfully")
        
        # Simple console interaction
        print("\n" + "="*50)
        print("🌱 Organic Farm Pest Management AI System")
        print("="*50)
        print("Console mode - Limited functionality")
        print("For full features, use: python start.py")
        print("="*50)
        
        while True:
            user_input = input("\nEnter 'test' to run system test, 'quit' to exit: ").strip().lower()
            
            if user_input == 'quit':
                break
            elif user_input == 'test':
                # Run a quick system test
                test_image = "test_images/aphids_high.jpg"
                if Path(test_image).exists():
                    print(f"Testing with {test_image}...")
                    result = system.identify_pest(test_image)
                    print(f"Result: {result}")
                else:
                    print("Test image not found")
            else:
                print("Unknown command. Try 'test' or 'quit'")
        
        logger.info("[STOP] Console session ended")
        
    except KeyboardInterrupt:
        logger.info("[STOP] Application stopped by user")
    except Exception as e:
        logger.error(f"[ERROR] Failed to start console interface: {str(e)}")
        return False
    
    return True

def package_setup():
    """
    Setup the system as a package (integrates setup.py functionality).
    """
    logger = logging.getLogger(__name__)
    
    logger.info("[SETUP] Configuring Organic Farm Pest Management AI System...")
    
    # Check if we're in development mode
    setup_py_path = Path("setup.py")
    if setup_py_path.exists():
        logger.info("[INFO] Development environment detected")
        
        # Install in development mode
        try:
            result = subprocess.run([
                sys.executable, 'setup.py', 'develop'
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                logger.info("[OK] Package setup complete - development mode")
                return True
            else:
                logger.warning(f"[WARN] Package setup failed: {result.stderr}")
        except Exception as e:
            logger.warning(f"[WARN] Package setup error: {e}")
    
    # Fallback: just ensure directories are properly configured
    logger.info("[INFO] Ensuring project structure...")
    
    required_dirs = ["models/optimized", "test_images", "docs"]
    for dir_path in required_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    logger.info("[OK] Basic package setup complete")
    return True

def install_enhanced_features():
    """Install enhanced features including ONNX for better optimization."""
    logger = logging.getLogger(__name__)
    
    pip_cmd = get_pip_command()
    if not pip_cmd:
        logger.error("[ERROR] pip not found")
        return False
    
    enhanced_packages = [
        'onnx>=1.14.0',
        'psutil>=5.9.0'
    ]
    
    logger.info("[INSTALL] Installing enhanced features...")
    
    for package in enhanced_packages:
        try:
            logger.info(f"[INSTALL] Installing {package.split('>=')[0]}...")
            result = subprocess.run(
                pip_cmd + ['install', package],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                logger.info(f"[OK] {package.split('>=')[0]} installed successfully")
            else:
                logger.warning(f"[WARN] Failed to install {package}: {result.stderr}")
                
        except Exception as e:
            logger.warning(f"[WARN] Error installing {package}: {e}")
    
    return True

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Organic Farm Pest Management AI System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python start.py                    # Launch web interface (default)
    python start.py --console          # Launch console interface  
    python start.py --setup            # Run setup and install dependencies
    python start.py --enhanced         # Install enhanced features (ONNX, psutil)
    python start.py --check            # Check dependencies only
    python start.py --package-setup    # Setup as Python package (development)
        """
    )
    
    parser.add_argument('--console', action='store_true', 
                       help='Launch console interface instead of web interface')
    parser.add_argument('--setup', action='store_true',
                       help='Run setup and dependency installation only')
    parser.add_argument('--check', action='store_true',
                       help='Check dependencies and system requirements only')
    parser.add_argument('--force-install', action='store_true',
                       help='Force reinstall all dependencies')
    parser.add_argument('--enhanced', action='store_true',
                       help='Install enhanced features (ONNX, psutil) for better performance')
    parser.add_argument('--package-setup', action='store_true',
                       help='Setup the system as a Python package (development mode)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    print("🌱 Organic Farm Pest Management AI System")
    print("="*50)
    
    # Always check Python version and system requirements
    if not check_python_version():
        sys.exit(1)
    
    if not check_system_requirements():
        sys.exit(1)
    
    # Check dependencies
    missing_packages, optional_missing = check_dependencies()
    
    if args.check:
        if missing_packages:
            logger.warning(f"[MISSING] Missing packages: {', '.join(missing_packages)}")
            logger.info("Run 'python start.py --setup' to install missing dependencies")
            sys.exit(1)
        elif optional_missing:
            logger.info(f"[OPTIONAL] Optional packages not installed: {', '.join(optional_missing)}")
            logger.info("Run 'python start.py --enhanced' to install enhanced features")
        else:
            logger.info("[OK] All dependencies are satisfied")
        sys.exit(0)
    
    # Package setup mode
    if args.package_setup:
        package_setup()
        sys.exit(0)
    
    # Enhanced features installation
    if args.enhanced:
        install_enhanced_features()
        logger.info("[OK] Enhanced features installation complete")
        sys.exit(0)
    
    # Install dependencies if needed or forced
    install_optional = args.enhanced or args.force_install
    if missing_packages or args.force_install:
        if not install_dependencies(missing_packages, install_optional):
            logger.error("[ERROR] Failed to install dependencies")
            sys.exit(1)
    
    if args.setup:
        # Also try to install enhanced features during setup
        if optional_missing:
            logger.info("[SETUP] Installing enhanced features for better performance...")
            install_enhanced_features()
        logger.info("[OK] Setup completed successfully")
        sys.exit(0)
    
    # Launch the appropriate interface
    if args.console:
        success = launch_console_interface()
    else:
        success = launch_web_interface()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
