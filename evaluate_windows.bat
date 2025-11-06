@echo off
REM Windows batch script to evaluate the model
REM Usage: evaluate_windows.bat <path_to_validation_data>

echo ===============================================================================
echo Model Evaluation Script for Windows
echo ===============================================================================
echo.

REM Check if data path provided
if "%1"=="" (
    echo ERROR: Please provide the path to your validation data
    echo.
    echo Usage:
    echo   evaluate_windows.bat "C:\path\to\validation\data"
    echo.
    echo Example:
    echo   evaluate_windows.bat "C:\Users\yaman\OneDrive\デスクトップ\modium\data\val"
    echo.
    echo Your validation data should have this structure:
    echo   validation_data\
    echo     benign\
    echo       image1.jpg
    echo       image2.jpg
    echo     suspicious\
    echo       image1.jpg
    echo     malignant\
    echo       image1.jpg
    echo.
    pause
    exit /b 1
)

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo [1/2] Installing required packages...
pip install gdown matplotlib seaborn scikit-learn tqdm

echo.
echo [2/2] Running evaluation...
python setup_and_evaluate.py --data "%1" --output reports\evaluation

if errorlevel 1 (
    echo.
    echo ERROR: Evaluation failed
    echo Please check the error messages above
    pause
    exit /b 1
)

echo.
echo ===============================================================================
echo SUCCESS! Evaluation completed
echo ===============================================================================
echo.
echo Results are saved in: reports\evaluation\
echo.
echo Open these files to view results:
echo   - reports\evaluation\confusion_matrix.png
echo   - reports\evaluation\roc_curves.png
echo   - reports\evaluation\precision_recall_curves.png
echo   - reports\evaluation\evaluation_results.json
echo.
echo To view detailed recommendations, read: MODEL_IMPROVEMENTS.md
echo.
pause
