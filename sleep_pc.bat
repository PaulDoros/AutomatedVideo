@echo off
echo Waiting for YouTube Shorts tasks to complete...

:: Wait 90 minutes (adjust as needed based on how long your tasks take)
timeout /t 5400 /nobreak

:: Check if any video processing is still happening
tasklist | findstr /i "ffmpeg" > nul
if %errorlevel% equ 0 (
    echo Video processing still running, waiting another 30 minutes...
    timeout /t 1800 /nobreak
)

:: Log that we're going to sleep
echo %date% %time% - Going to sleep after completing YouTube Shorts tasks >> sleep_log.txt

:: Put the PC to sleep
rundll32.exe powrprof.dll,SetSuspendState 0,1,0

echo Done! 