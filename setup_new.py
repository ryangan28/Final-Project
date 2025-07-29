"""
Package setup configuration for Organic Farm Pest Management AI System
Standard setuptools configuration for package installation.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
requirements_file = this_directory / "requirements.txt"
if requirements_file.exists():
    requirements = requirements_file.read_text().strip().split('\n')
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]

setup(
    name="organic-farm-pest-management",
    version="1.0.0",
    author="Ryan Gan",
    author_email="your.email@example.com",
    description="AI-powered organic farm pest management system with offline-first capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ryangan28/Final-Project",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "demo": [
            "streamlit>=1.28.0",
            "Pillow>=9.5.0",
            "numpy>=1.24.0",
            "pathlib2>=2.3.7",
        ],
        "full": [
            "torch>=2.0.0",
            "torchvision>=0.15.0", 
            "opencv-python>=4.8.0",
            "scikit-learn>=1.3.0",
            "onnx>=1.14.0",
            "onnxruntime>=1.15.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "pest-management=start:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.jpg", "*.png", "*.md", "*.txt"],
    },
    project_urls={
        "Bug Reports": "https://github.com/ryangan28/Final-Project/issues",
        "Source": "https://github.com/ryangan28/Final-Project",
        "Documentation": "https://github.com/ryangan28/Final-Project/blob/main/README.md",
    },
)
