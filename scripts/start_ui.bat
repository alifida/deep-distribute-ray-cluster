@echo off
setlocal
set SCRIPT_DIR=%~dp0
powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%start_ui.ps1"
set EXIT_CODE=%ERRORLEVEL%
echo.
if not "%EXIT_CODE%"=="0" (
  echo [start_ui.bat] FAILED with exit code %EXIT_CODE%.
) else (
  echo [start_ui.bat] Completed.
)
echo Press any key to close this window...
pause >nul
endlocal
