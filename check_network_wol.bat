@echo off
echo Checking network adapters for Wake-on-LAN support...

:: List all network adapters
powercfg /devicequery wake_armed

echo.
echo The devices listed above can currently wake your computer.
echo Your Wi-Fi adapter with MAC address 70-D8-23-5C-3F-64 should be in the list.
echo.
echo If you don't see your network adapter, run the following in an admin command prompt:
echo powercfg /deviceenablewake "Your Network Adapter Name"
echo.
echo You can find your exact adapter name in Device Manager.
echo.
pause 