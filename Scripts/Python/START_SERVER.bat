@echo off
echo ========================================
echo FOOTBALL JERSEY FILTER SERVER
echo ========================================
echo Brighton Home Jersey - ML Filter
echo Press Ctrl+C to stop
echo ========================================
echo.
cd /d "%~dp0"
python jersey_filter_server.py
pause
