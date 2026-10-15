@echo off
rem Builds the concilium image once (10-20 min, ~2 GB): every package is pinned by hash, so all parties get the same.
cd /d "%~dp0"
if not exist Dockerfile (echo Files missing. Right-click the zip, choose "Extract All", then run build.cmd in the extracted folder. & pause & exit /b 1)
docker build --target tool -t concilium . || (echo Build failed. Is Docker Desktop running? & pause & exit /b 1)
call concilium.cmd version
echo Done. Compare the code hash above with the other parties.
pause
