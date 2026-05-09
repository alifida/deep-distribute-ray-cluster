@echo off
setlocal
set SCRIPT_DIR=%~dp0
set HEAD_IP=%~1
set HEAD_PORT=%~2
if "%HEAD_IP%"=="" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%start_worker.ps1"
) else (
  if "%HEAD_PORT%"=="" set HEAD_PORT=6379
  powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%start_worker.ps1" -HeadIp "%HEAD_IP%" -HeadPort "%HEAD_PORT%"
)
set EXIT_CODE=%ERRORLEVEL%
echo.
if not "%EXIT_CODE%"=="0" (
  echo [start_worker.bat] FAILED with exit code %EXIT_CODE%.
  echo Check HEAD_IP in scripts\cluster_config.env or pass HEAD_IP explicitly.
) else (
  echo [start_worker.bat] Completed. If worker is not visible on head, verify firewall and HEAD_IP.
)
echo Press any key to close this window...
pause >nul
endlocal
