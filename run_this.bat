@echo off
REM Navigate to project folder if needed
cd /d %~dp0

REM Activate virtual environment
call web\\Scripts\\activate.bat

REM Run your Python app
python app.py

REM Keep window open
pause
