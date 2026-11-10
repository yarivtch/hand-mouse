@echo off
cd /d "%~dp0"
py -m pip install --quiet -r requirements.txt
py hand_mouse.py %*
pause
