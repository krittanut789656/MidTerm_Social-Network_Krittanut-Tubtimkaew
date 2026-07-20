@echo off
echo ================================================
echo  GNN Final Project Pipeline
echo  Thai Bank Graph Analysis
echo ================================================
echo.

:: ไปที่ project directory
cd /d "%~dp0"
echo [1/5] Working directory: %cd%
echo.

:: ลอง activate conda base ก่อน
call conda activate base 2>nul
if %errorlevel% neq 0 (
    echo conda not found, using system python...
)

:: ตรวจสอบ Python
python --version
echo.

:: ติดตั้ง dependencies
echo [2/5] Installing PyTorch Geometric and Captum...
echo (This may take a few minutes the first time)
echo.

pip install torch --quiet
pip install torch_geometric --quiet
pip install captum --quiet
pip install scikit-learn --quiet

echo.
echo [3/5] Building Graph Snapshot Dataset...
python src/gnn_dataset.py
if %errorlevel% neq 0 (
    echo ERROR in gnn_dataset.py
    pause
    exit /b 1
)

echo.
echo [4/5] Training GNN Model...
python src/gnn_train.py
if %errorlevel% neq 0 (
    echo ERROR in gnn_train.py
    pause
    exit /b 1
)

echo.
echo [5/5] Computing Captum Edge Attributions...
python src/gnn_explain.py
if %errorlevel% neq 0 (
    echo ERROR in gnn_explain.py
    pause
    exit /b 1
)

echo.
echo ================================================
echo  Pipeline Complete!
echo  Now open Streamlit to see Page 10
echo ================================================
echo.
pause
