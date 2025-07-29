@echo off
echo Starting Organic Farm Pest Management AI System Web Interface...
echo.
cd /d "%~dp0"
echo. | python -m streamlit run streamlit_app.py
pause
