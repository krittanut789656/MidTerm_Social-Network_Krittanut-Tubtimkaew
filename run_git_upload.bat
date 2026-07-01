@echo off
cd /d "%~dp0"
echo === Thai Bank Graph Analysis - GitHub Upload ===
echo.
echo [1/5] git init...
git init -b main
git config user.name "Krittanut Tubtimkaew"
git config user.email "kt.booklover5555@gmail.com"
echo.
echo [2/5] git add all files...
git add .
echo.
echo [3/5] git status...
git status --short
echo.
echo [4/5] git commit...
git commit -m "Initial commit: Thai Bank Graph Analysis - Social Network Midterm Project"
echo.
echo [5/5] Setting remote and pushing...
git remote add origin https://github.com/krittanut789656/MidTerm_Social-Network_Krittanut-Tubtimkaew.git
git push -u origin main
echo.
echo === Done! Check GitHub repo ===
pause
