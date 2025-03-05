@echo off
echo Disabling lock screen when waking from sleep...

:: Disable the lock screen when resuming from sleep
reg add "HKEY_LOCAL_MACHINE\SOFTWARE\Policies\Microsoft\Windows\Personalization" /v NoLockScreen /t REG_DWORD /d 1 /f

:: Disable requiring sign-in when waking from sleep
reg add "HKEY_LOCAL_MACHINE\SOFTWARE\Policies\Microsoft\Power\PowerSettings\0e796bdb-100d-47d6-a2d5-f7d2daa51f51" /v ACSettingIndex /t REG_DWORD /d 0 /f
reg add "HKEY_LOCAL_MACHINE\SOFTWARE\Policies\Microsoft\Power\PowerSettings\0e796bdb-100d-47d6-a2d5-f7d2daa51f51" /v DCSettingIndex /t REG_DWORD /d 0 /f

echo Done! The PC should no longer require a PIN or password when waking from sleep.
echo You may need to restart your computer for these changes to take effect.
pause 