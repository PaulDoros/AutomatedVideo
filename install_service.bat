@echo off
echo Installing Python requirements...
pip install schedule pywin32

echo Installing the YouTube Shorts Wake-on-LAN service...
python install_wol_service.py install

echo Starting the service...
python install_wol_service.py start

echo Done! You can check Windows Services to verify it's running.
echo (Press any key to exit)
pause 