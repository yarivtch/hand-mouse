@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Creating a virtual environment in .venv - first run only...
    py -m venv .venv || goto :fail
)
".venv\Scripts\python.exe" -m pip install --quiet -r requirements.txt || goto :fail
".venv\Scripts\python.exe" paper_board.py %*
pause
exit /b

:fail
echo.
echo Setup failed. Install Python 3.10-3.12 from python.org, then run this file again.
pause
exit /b 1
