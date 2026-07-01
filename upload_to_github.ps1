# ============================================================
# upload_to_github.ps1
# รันใน PowerShell ที่โฟลเดอร์โปรเจกต์
# Double-click หรือ: Right-click -> Run with PowerShell
# ============================================================

$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectDir

Write-Host ""
Write-Host "=== Thai Bank Graph Analysis — GitHub Upload ===" -ForegroundColor Cyan
Write-Host "Project folder: $projectDir"
Write-Host ""

# 1. ลบ .git ที่ค้างไว้ (ถ้ามี)
if (Test-Path ".git") {
    Write-Host "[1/6] ลบ .git เก่าที่ค้างไว้..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force ".git"
}

# 2. git init + config
Write-Host "[2/6] กำลัง git init..." -ForegroundColor Yellow
git init -b main
git config user.name "Krittanut Tubtimkaew"
git config user.email "kt.booklover5555@gmail.com"

# 3. ตรวจว่า .gitignore มีอยู่
if (-not (Test-Path ".gitignore")) {
    Write-Host "[3/6] สร้าง .gitignore..." -ForegroundColor Yellow
    @"
# Secrets
.env
.env.*
*.env

# Python cache
__pycache__/
*.pyc
*.pyo

# Virtual environments
venv/
.venv/

# Jupyter checkpoints
.ipynb_checkpoints/

# OS / IDE
.DS_Store
Thumbs.db
.vscode/
.idea/
"@ | Set-Content ".gitignore"
} else {
    Write-Host "[3/6] .gitignore มีอยู่แล้ว OK" -ForegroundColor Green
}

# 4. git add ทุกไฟล์
Write-Host "[4/6] กำลัง git add..." -ForegroundColor Yellow
git add .

# แสดงไฟล์ที่จะ commit
Write-Host ""
Write-Host "ไฟล์ที่จะ commit:" -ForegroundColor Cyan
git status --short
Write-Host ""

# 5. git commit
Write-Host "[5/6] กำลัง git commit..." -ForegroundColor Yellow
git commit -m "Initial commit: Thai Bank Graph Analysis — Social Network Midterm Project

- 18 nodes: 7 Thai banks + 11 global macro/FX/ETF factors
- 60 validated edges (Pearson + FDR correction)
- 14 partial correlation edges (GraphicalLassoCV)
- 9 OLS factor exposure edges
- Louvain community detection: 4 communities
- Regime analysis: Hiking vs Cutting period comparison
- Streamlit dashboard: 9 pages
- Automated validation: 66/66 checks PASS"

# 6. set remote + push
Write-Host "[6/6] กำลังตั้ง remote และ push..." -ForegroundColor Yellow
git remote add origin https://github.com/krittanut789656/MidTerm_Social-Network_Krittanut-Tubtimkaew.git
git push -u origin main

Write-Host ""
Write-Host "=== Upload สำเร็จ! ===" -ForegroundColor Green
Write-Host "ดูผลได้ที่: https://github.com/krittanut789656/MidTerm_Social-Network_Krittanut-Tubtimkaew" -ForegroundColor Cyan
Write-Host ""
