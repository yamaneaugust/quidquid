@echo off
REM Windows batch script to set up ISIC dataset
REM Run this after downloading ISIC2018_Task3_Training_Input.zip and CSV

echo ========================================
echo ISIC Dataset Setup for Windows
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from python.org
    pause
    exit /b 1
)

echo Step 1: Activating virtual environment...
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    echo ERROR: Virtual environment not found!
    echo Please run: python -m venv venv
    pause
    exit /b 1
)

echo.
echo Step 2: Extracting training images...
echo Please ensure ISIC2018_Task3_Training_Input.zip is in your Downloads folder
set DOWNLOADS=%USERPROFILE%\Downloads
set ZIP_FILE=%DOWNLOADS%\ISIC2018_Task3_Training_Input.zip
set EXTRACT_DIR=%DOWNLOADS%\extracted

if exist "%ZIP_FILE%" (
    python scripts/organize_isic_data.py --extract "%ZIP_FILE%" --extract-to "%EXTRACT_DIR%"
) else (
    echo WARNING: %ZIP_FILE% not found
    echo Please download it from: https://challenge.isic-archive.com/data/
    echo.
)

echo.
echo Step 3: Organizing images into class folders...
set IMAGES_DIR=%EXTRACT_DIR%\ISIC2018_Task3_Training_Input
set CSV_FILE=%DOWNLOADS%\ISIC2018_Task3_Training_GroundTruth.csv

if exist "%CSV_FILE%" (
    python scripts/organize_isic_data.py --images-dir "%IMAGES_DIR%" --groundtruth-csv "%CSV_FILE%" --output-dir data\raw --task3-format
) else (
    echo WARNING: %CSV_FILE% not found
    echo Please download it from: https://challenge.isic-archive.com/data/
    echo.
)

echo.
echo Step 4: (Optional) Simplify to 3 classes...
set /p SIMPLIFY="Do you want to simplify from 7 to 3 classes? (y/n): "
if /i "%SIMPLIFY%"=="y" (
    python scripts/simplify_classes.py --source data\raw --target data\raw_simplified --copy
    echo.
    echo Dataset simplified! You can now train with:
    echo   python -m src.model.train --data-dir data\raw_simplified --num-classes 3
) else (
    echo.
    echo You can train with all 7 classes:
    echo   python -m src.model.train --data-dir data\raw --num-classes 7
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo   1. Verify data in data\raw\ folder
echo   2. Train your model (see commands above)
echo   3. Run inference with: python -m src.inference.predict
echo.

pause
