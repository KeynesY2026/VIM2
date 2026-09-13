@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python 3.10 through 3.13 is required but was not found.
    pause
    exit /b 1
)
set "PYTHONNOUSERSITE=1"
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "HF_HOME=%~dp0.runtime\huggingface"
set "PYTHONPATH=%~dp0app;%~dp0runtime\site-packages"

python -m vim2
if errorlevel 1 (
    echo.
    echo VIM2 stopped because startup or runtime validation failed.
    pause
    exit /b 1
)
