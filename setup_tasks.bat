@echo off
echo Setting up Windows Task Scheduler for YouTube Shorts automation...

:: Create directory for scripts if it doesn't exist
mkdir "%USERPROFILE%\YouTubeShortsWoL" 2>nul

:: Copy files to the directory
echo Copying files to %USERPROFILE%\YouTubeShortsWoL...
copy router_wol.py "%USERPROFILE%\YouTubeShortsWoL\"
copy sleep_pc.bat "%USERPROFILE%\YouTubeShortsWoL\"
copy test_wol.bat "%USERPROFILE%\YouTubeShortsWoL\"

:: Get current directory for the tasks
set SCRIPT_DIR=%USERPROFILE%\YouTubeShortsWoL

:: Create scheduled tasks for each time
echo Creating scheduled tasks...

:: Business Channel - 09:30
schtasks /create /tn "YouTube Shorts - Business Wake" /tr "pythonw.exe \"%SCRIPT_DIR%\router_wol.py\" --test" /sc daily /st 09:30:00 /ru "%USERNAME%" /F

:: Tech Humor Channel - 13:30
schtasks /create /tn "YouTube Shorts - Tech Humor 1 Wake" /tr "pythonw.exe \"%SCRIPT_DIR%\router_wol.py\" --test" /sc daily /st 13:30:00 /ru "%USERNAME%" /F

:: Tech Channel - 14:30
schtasks /create /tn "YouTube Shorts - Tech Wake" /tr "pythonw.exe \"%SCRIPT_DIR%\router_wol.py\" --test" /sc daily /st 14:30:00 /ru "%USERNAME%" /F

:: Business Channel 2 - 15:30
schtasks /create /tn "YouTube Shorts - Business 2 Wake" /tr "pythonw.exe \"%SCRIPT_DIR%\router_wol.py\" --test" /sc daily /st 15:30:00 /ru "%USERNAME%" /F

:: Tech Humor Channel 2 - 17:30
schtasks /create /tn "YouTube Shorts - Tech Humor 2 Wake" /tr "pythonw.exe \"%SCRIPT_DIR%\router_wol.py\" --test" /sc daily /st 17:30:00 /ru "%USERNAME%" /F

:: Gaming Channel - 19:30
schtasks /create /tn "YouTube Shorts - Gaming Wake" /tr "pythonw.exe \"%SCRIPT_DIR%\router_wol.py\" --test" /sc daily /st 19:30:00 /ru "%USERNAME%" /F

:: Daily Schedule Generation - 00:15
schtasks /create /tn "YouTube Shorts - Schedule Gen Wake" /tr "pythonw.exe \"%SCRIPT_DIR%\router_wol.py\" --test" /sc daily /st 00:15:00 /ru "%USERNAME%" /F

:: Performance Analysis - 01:45
schtasks /create /tn "YouTube Shorts - Analysis Wake" /tr "pythonw.exe \"%SCRIPT_DIR%\router_wol.py\" --test" /sc daily /st 01:45:00 /ru "%USERNAME%" /F

:: Sleep tasks (after each primary task)
schtasks /create /tn "YouTube Shorts - Business Sleep" /tr "\"%SCRIPT_DIR%\sleep_pc.bat\"" /sc daily /st 10:30:00 /ru "%USERNAME%" /F
schtasks /create /tn "YouTube Shorts - Tech Humor 1 Sleep" /tr "\"%SCRIPT_DIR%\sleep_pc.bat\"" /sc daily /st 14:30:00 /ru "%USERNAME%" /F
schtasks /create /tn "YouTube Shorts - Tech Sleep" /tr "\"%SCRIPT_DIR%\sleep_pc.bat\"" /sc daily /st 15:30:00 /ru "%USERNAME%" /F
schtasks /create /tn "YouTube Shorts - Business 2 Sleep" /tr "\"%SCRIPT_DIR%\sleep_pc.bat\"" /sc daily /st 16:30:00 /ru "%USERNAME%" /F
schtasks /create /tn "YouTube Shorts - Tech Humor 2 Sleep" /tr "\"%SCRIPT_DIR%\sleep_pc.bat\"" /sc daily /st 18:30:00 /ru "%USERNAME%" /F
schtasks /create /tn "YouTube Shorts - Gaming Sleep" /tr "\"%SCRIPT_DIR%\sleep_pc.bat\"" /sc daily /st 20:30:00 /ru "%USERNAME%" /F
schtasks /create /tn "YouTube Shorts - Schedule Gen Sleep" /tr "\"%SCRIPT_DIR%\sleep_pc.bat\"" /sc daily /st 01:15:00 /ru "%USERNAME%" /F
schtasks /create /tn "YouTube Shorts - Analysis Sleep" /tr "\"%SCRIPT_DIR%\sleep_pc.bat\"" /sc daily /st 02:45:00 /ru "%USERNAME%" /F

echo All tasks created successfully!
echo Your PC will now automatically wake up for YouTube Shorts tasks and go back to sleep afterward.
pause 