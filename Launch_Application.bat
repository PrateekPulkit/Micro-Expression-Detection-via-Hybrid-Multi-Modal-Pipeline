@echo off
title Micro-Expression Detection - Real-time App
echo ============================================================
echo   MICRO-EXPRESSION DETECTION - REAL-TIME APPLICATION
echo ============================================================
echo.
echo [INFO] Device: Auto-detecting...
echo [INFO] Model: Hybrid-TCN (Full Variant)
echo [INFO] Checkpoint: checkpoints/best_fold0_full.pt
echo.
echo [HINT] Press 'Q' to quit the application window.
echo [HINT] Ensure your face is well-lit for best results.
echo.
echo ============================================================
echo.

:: Check for model setup
python scripts/setup_models.py

echo.
echo [STATUS] Launching Application...
echo.

python -m inference.realtime_inference --config configs/config.yaml --checkpoint checkpoints/best_fold0_full.pt

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Application failed to launch. 
    echo Please ensure your webcam is connected and no other app is using it.
    pause
)
