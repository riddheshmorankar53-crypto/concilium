@echo off
rem One concilium command, run in Docker on this folder: concilium keys|config|board|member|participant|version ...
rem Your keys, the config, your data and task file stay in this folder; the container sees only this folder.
cd /d "%~dp0"
set PORTARG=
if not defined PORT set PORT=8000
if /i "%1"=="board" set PORTARG=-p %PORT%:%PORT%
docker run --rm %PORTARG% -e TASK -e NET_TIMEOUT -e PORT -e OMP_WAIT_POLICY=PASSIVE -v "%cd%:/work" concilium concilium %*
