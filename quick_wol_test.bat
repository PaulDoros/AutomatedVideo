@echo off
echo Wake-on-LAN Quick Test

:: Create a timestamp 20 seconds from now (giving us time to fall asleep)
for /f "tokens=1-3 delims=:." %%a in ("%time%") do (
    set /a hour=%%a
    set /a minute=%%b
    set /a second=%%c+20
)
if %second% geq 60 (
    set /a second-=60
    set /a minute+=1
)
if %minute% geq 60 (
    set /a minute-=60
    set /a hour+=1
)
if %hour% geq 24 (
    set /a hour-=24
)

:: Format the numbers to have leading zeros if needed
if %hour% lss 10 set hour=0%hour%
if %minute% lss 10 set minute=0%minute%
if %second% lss 10 set second=0%second%

:: Set the time for the wake task
set waketime=%hour%:%minute%:%second%

echo Creating a scheduled task to wake up at %waketime%

:: Create a temporary task that will run once 20 seconds from now
schtasks /create /tn "WOL_Quick_Test" /tr "pythonw.exe \"%CD%\router_wol.py\" --test" /sc once /st %waketime% /ru "%USERNAME%" /F

echo Your PC will go to sleep in 5 seconds and wake up at %waketime%
echo Please save any important work before proceeding!
timeout /t 5

:: Put the PC to sleep
rundll32.exe powrprof.dll,SetSuspendState 0,1,0

:: This will only execute after waking up
echo If you can see this message, your PC successfully woke up!
echo Cleaning up temporary wake task...
schtasks /delete /tn "WOL_Quick_Test" /f
pause 