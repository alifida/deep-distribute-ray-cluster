@echo off
setlocal
set SCRIPT_DIR=%~dp0
set TARGET_IP=%~1
set MODE=%~2
if "%MODE%"=="" set MODE=full
if "%TARGET_IP%"=="" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%check_ports.ps1" -Mode "%MODE%"
) else (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%check_ports.ps1" -TargetIp "%TARGET_IP%" -Mode "%MODE%"
)
endlocal
