@echo off
echo Enabling Wake Timers in Windows Power Settings...

:: Enable wake timers for both AC and DC power
powercfg /SETACVALUEINDEX SCHEME_CURRENT SUB_SLEEP RTCWAKE 1
powercfg /SETDCVALUEINDEX SCHEME_CURRENT SUB_SLEEP RTCWAKE 1

:: Enable wake-up from devices (for network adapters)
powercfg /SETACVALUEINDEX SCHEME_CURRENT SUB_SLEEP ACALLOWHYBRIDWAKE 1
powercfg /SETDCVALUEINDEX SCHEME_CURRENT SUB_SLEEP ACALLOWHYBRIDWAKE 1

:: Save changes
powercfg /SETACTIVE SCHEME_CURRENT

echo Wake Timers have been enabled!
echo Your PC can now wake up from scheduled tasks.
pause 