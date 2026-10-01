@echo off
setlocal EnableDelayedExpansion
:: universr.bat — install (if needed) then run an UniverSR command or drop into a shell

set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "VENV_DIR=%SCRIPT_DIR%\.venv"

:: 0. Pull latest changes if this is a git checkout
if exist "%SCRIPT_DIR%\.git" (
    where git >nul 2>&1
    if !errorlevel! equ 0 (
        echo [universr] Checking for updates...
        pushd "%SCRIPT_DIR%"
        git pull --ff-only
        if !errorlevel! neq 0 echo [universr] WARNING: git pull failed -- continuing with local copy.
        popd
    )
)

:: 1. Locate or install uv
set "UV_BIN="
where uv >nul 2>&1 && for /f "delims=" %%i in ('where uv') do set "UV_BIN=%%i" & goto :uv_found
if exist "%USERPROFILE%\.local\bin\uv.exe"  set "UV_BIN=%USERPROFILE%\.local\bin\uv.exe" & goto :uv_found
if exist "%USERPROFILE%\.cargo\bin\uv.exe"  set "UV_BIN=%USERPROFILE%\.cargo\bin\uv.exe" & goto :uv_found
if exist "%LOCALAPPDATA%\uv\uv.exe"         set "UV_BIN=%LOCALAPPDATA%\uv\uv.exe"        & goto :uv_found

echo [universr] uv not found -- installing...
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
if %errorlevel% neq 0 ( echo [universr] ERROR: Could not install uv. & pause & exit /b 1 )
if exist "%USERPROFILE%\.local\bin\uv.exe" set "UV_BIN=%USERPROFILE%\.local\bin\uv.exe" & goto :uv_found
if exist "%LOCALAPPDATA%\uv\uv.exe"        set "UV_BIN=%LOCALAPPDATA%\uv\uv.exe"        & goto :uv_found
where uv >nul 2>&1 && for /f "delims=" %%i in ('where uv') do set "UV_BIN=%%i" & goto :uv_found
echo [universr] ERROR: uv installed but not found. Restart terminal and retry.
pause & exit /b 1

:uv_found
echo [universr] uv: %UV_BIN%

:: 2. Create venv if missing
if exist "%VENV_DIR%\Scripts\python.exe" (
    echo [universr] venv exists
) else (
    echo [universr] Creating virtual environment (Python 3.11^)...
    "%UV_BIN%" venv "%VENV_DIR%" --python 3.11
    if %errorlevel% neq 0 ( echo [universr] ERROR: venv creation failed. & pause & exit /b 1 )
)

:: 3. Install / sync dependencies
echo [universr] Installing PyTorch (CUDA 12.6^)...
"%UV_BIN%" pip install ^
    --python "%VENV_DIR%\Scripts\python.exe" ^
    --index-url https://download.pytorch.org/whl/cu126 ^
    torch==2.7.1+cu126 torchaudio==2.7.1+cu126
if %errorlevel% neq 0 ( echo [universr] ERROR: PyTorch install failed. & pause & exit /b 1 )
echo [universr] PyTorch installed

echo [universr] Syncing remaining dependencies...
"%UV_BIN%" pip install ^
    --python "%VENV_DIR%\Scripts\python.exe" ^
    "setuptools<71" pyyaml ^
    -r "%SCRIPT_DIR%\requirements.txt"
if %errorlevel% neq 0 ( echo [universr] ERROR: Dependency install failed. & pause & exit /b 1 )
echo [universr] Dependencies up to date

:: 4. Strip --dev flag and set env var if present
set "UNIVERSR_DEV="
set "FILTERED_ARGS="
set "_RAW_ARGS=%*"
if defined _RAW_ARGS (
    for %%a in (%*) do (
        if /i "%%a"=="--dev" (
            set "UNIVERSR_DEV=1"
        ) else (
            set "FILTERED_ARGS=!FILTERED_ARGS! %%a"
        )
    )
    if defined FILTERED_ARGS set "FILTERED_ARGS=!FILTERED_ARGS:~1!"
)

:: 5. If arguments given, treat first as the script name and run it
if not "!FILTERED_ARGS!"=="" (
    for /f "tokens=1,* delims= " %%a in ("!FILTERED_ARGS!") do set "CMD=%%a" & set "REST=%%b"
    if /i "!CMD!"=="train"     set "SCRIPT=train.py"
    if /i "!CMD!"=="inference" set "SCRIPT=inference.py"
    if /i "!CMD!"=="test"      set "SCRIPT=test.py"
    if defined SCRIPT (
        "%VENV_DIR%\Scripts\python.exe" "%SCRIPT_DIR%\!SCRIPT!" !REST!
    ) else if /i "!CMD!"=="python" (
        "%VENV_DIR%\Scripts\python.exe" !REST!
    ) else (
        "%VENV_DIR%\Scripts\python.exe" "%SCRIPT_DIR%\!CMD!" !REST!
    )
    exit /b %errorlevel%
)

:: 6. No arguments — launch TUI
"%VENV_DIR%\Scripts\python.exe" "%SCRIPT_DIR%\utils\tui.py"