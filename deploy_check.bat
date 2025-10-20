@echo off
echo ===================================
echo  POTHOLE DETECTION - DEPLOY HELPER
echo ===================================
echo.

echo This script will help you prepare for deployment.
echo.

REM Check if best_02.pt exists in backend folder
if exist "best_02.pt" (
    echo [OK] Model file found: best_02.pt
) else (
    echo [WARNING] Model file NOT found in backend folder!
    echo.
    echo Please copy best_02.pt from:
    echo   ..\..\best_02.pt
    echo To:
    echo   %CD%\best_02.pt
    echo.
    pause
    exit /b 1
)

echo.
echo ===================================
echo  DEPLOYMENT CHECKLIST
echo ===================================
echo.
echo [ ] 1. Model file copied to backend folder (best_02.pt)
echo [ ] 2. Created GitHub repository for backend
echo [ ] 3. Pushed code to GitHub
echo [ ] 4. Created account on Render.com
echo [ ] 5. Deployed backend to Render
echo [ ] 6. Updated api_service.dart with production URL
echo [ ] 7. Rebuilt Flutter APK
echo [ ] 8. Tested app with production backend
echo.
echo See DEPLOYMENT_GUIDE.md for detailed instructions!
echo.
pause
