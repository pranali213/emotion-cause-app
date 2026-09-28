@echo off
echo ============================================
echo   ECPE AI Lab - Starting...
echo ============================================
pip install -r requirements.txt -q
echo.
echo Open http://localhost:5000 in your browser
echo Press Ctrl+C to stop
echo.
python app.py
pause
