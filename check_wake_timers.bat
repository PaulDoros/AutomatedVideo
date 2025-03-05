@echo off
echo Checking if Wake Timers are enabled...

:: Check AC power wake timers
powercfg /q SCHEME_CURRENT SUB_SLEEP RTCWAKE
echo.
echo If you see "Current AC Power Setting Index: 0x00000001" above, 
echo wake timers are ENABLED for AC power.
echo.

:: Check if wake from device is allowed (needed for network adapters)
powercfg /q SCHEME_CURRENT SUB_SLEEP ACALLOWHYBRIDWAKE
echo.
echo If you see "Current AC Power Setting Index: 0x00000001" above,
echo wake-up from devices is ENABLED.
echo.

echo Make sure to run enable_wake_timers.bat if these settings are not enabled.
pause 