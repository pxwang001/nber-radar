@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo First run: creating Python environment...
    py -3 -m venv .venv
    if errorlevel 1 goto :error
)

call ".venv\Scripts\activate.bat"
python -m pip install -r requirements.txt
if errorlevel 1 goto :error

python -m streamlit run app.py
exit /b 0

:error
echo.
echo Start failed. Please check that Python 3 is installed.
pause
exit /b 1
