@echo off
cd /d "%~dp0"
py -3 run_all.py
if errorlevel 1 exit /b 1
py -3 verify_results.py
if errorlevel 1 exit /b 1
py -3 -m unittest discover -s tests
if errorlevel 1 exit /b 1
py -3 verify_checksums.py
