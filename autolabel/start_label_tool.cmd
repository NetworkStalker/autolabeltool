@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" label_tool.py %*
    if errorlevel 1 pause
    exit /b
)
set "bundled=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if exist ".deps\PySide6" if exist ".deps\yaml" if exist "%bundled%" (
    set "PYTHONPATH=%CD%\.deps"
    "%bundled%" label_tool.py %*
    if errorlevel 1 pause
    exit /b
)
python label_tool.py %*
if errorlevel 1 (
    echo Follow the installation steps in README.md.
    pause
)

