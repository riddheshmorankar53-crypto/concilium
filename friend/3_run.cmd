@echo off
cd /d "%~dp0"
if not exist concilium.json (echo Put the concilium.json your friend sent into this folder first. & pause & exit /b 1)
set /p IP=Paste the address your friend sent (e.g. 192.168.1.2, or 100.x.y.z with Tailscale):
docker rm -f concilium-member >nul 2>&1
docker run --rm --name concilium-member --add-host concilium-board:%IP% -e NET_TIMEOUT=60 -e OMP_NUM_THREADS=4 -e OMP_WAIT_POLICY=PASSIVE -v "%cd%:/work" concilium concilium member concilium.json m1.key
if errorlevel 1 (echo The run stopped with an error. Send your friend a screenshot of this window.) else (echo The run finished. Send your friend a screenshot of this window.)
pause
