@echo off
echo Testing Wake-on-LAN functionality...
python router_wol.py --test
echo Test complete!
timeout /t 5 