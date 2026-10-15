@echo off
cd /d "%~dp0"
if not exist Dockerfile (echo Files missing. Right-click the zip, choose "Extract All", then double-click 1_build.cmd in the extracted folder. & pause & exit /b 1)
docker build --target tool -t concilium . || (echo Build failed. Is Docker Desktop running? & pause & exit /b 1)
echo Done.
pause
