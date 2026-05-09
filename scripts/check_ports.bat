@echo off
setlocal
if "%~1"=="" (
  echo Usage: %~nx0 ^<TARGET_IP^> [quick^|full]
  exit /b 1
)
set SCRIPT_DIR=%~dp0
set TARGET_IP=%~1
set MODE=%~2
if "%MODE%"=="" set MODE=full
powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%check_ports.ps1" -TargetIp "%TARGET_IP%" -Mode "%MODE%"
endlocal
