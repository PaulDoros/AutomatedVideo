@echo off
echo Setting up automatic login after wake-up...

:: Create directory if it doesn't exist
mkdir "%USERPROFILE%\YouTubeShortsWoL" 2>nul

:: Copy the PowerShell script
copy "auto_login.ps1" "%USERPROFILE%\YouTubeShortsWoL\" /Y

:: Create a scheduled task to run the script at logon
schtasks /create /tn "Auto Login After Wake" /tr "powershell.exe -ExecutionPolicy Bypass -File \"%USERPROFILE%\YouTubeShortsWoL\auto_login.ps1\"" /sc onlogon /ru "%USERNAME%" /F

echo.
echo Done! The automatic login has been set up.
echo.
echo IMPORTANT SECURITY NOTE:
echo This script will automatically enter your PIN (3775) when your PC wakes up.
echo Anyone with physical access to your PC will be able to bypass the lock screen.
echo.
echo For better security, consider using Option 1 (disable lock screen after sleep)
echo by running the disable_lock_on_wake.bat script as administrator.
pause 