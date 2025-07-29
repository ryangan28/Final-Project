"""
Streamlit Web Interface for Organic Farm Pest Management AI System
Run this file with: python -m streamlit run streamlit_app.py
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the main system
from main import PestManagementSystem
from mobile.app_interface import create_app

# Initialize the system
if 'pest_system' not in st.session_state:
    st.session_state.pest_system = PestManagementSystem()

# Create and run the app
app = create_app(st.session_state.pest_system)
app.run()
