@echo off
cd /d "%~dp0"

:: Use GitHub Desktop's bundled git
set GIT="C:\Users\USER\AppData\Local\GitHubDesktop\app-3.5.12\resources\app\git\cmd\git.exe"

echo === Adding remote and pushing to GitHub ===

:: Add remote (ignore error if already exists)
%GIT% remote add origin https://github.com/krittanut789656/MidTerm_Social-Network_Krittanut-Tubtimkaew.git 2>nul

:: Push
%GIT% push -u origin main

echo.
echo === Done! Check https://github.com/krittanut789656/MidTerm_Social-Network_Krittanut-Tubtimkaew ===
pause
