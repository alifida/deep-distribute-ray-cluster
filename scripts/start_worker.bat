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
endlocal
