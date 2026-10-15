@echo off
cd /d "%~dp0"
if exist m1.key (echo m1.key already exists: keeping it. & pause & exit /b 0)
docker run --rm -v "%cd%:/work" concilium concilium keys m1 || (echo Failed. Did 1_build finish? & pause & exit /b 1)
echo.
echo Send ONLY the file m1.pub to your friend. Never send m1.key.
pause
