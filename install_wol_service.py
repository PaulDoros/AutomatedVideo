import os
import sys
import win32serviceutil
import win32service
import win32event
import servicemanager
import socket
import time
import subprocess

class WoLService(win32serviceutil.ServiceFramework):
    _svc_name_ = "YouTubeShortsWoL"
    _svc_display_name_ = "YouTube Shorts Wake-on-LAN Service"
    _svc_description_ = "Sends scheduled Wake-on-LAN packets for YouTube Shorts automation"

    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
        socket.setdefaulttimeout(60)
        self.is_running = True

    def SvcStop(self):
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.hWaitStop)
        self.is_running = False

    def SvcDoRun(self):
        servicemanager.LogMsg(servicemanager.EVENTLOG_INFORMATION_TYPE,
                              servicemanager.PYS_SERVICE_STARTED,
                              (self._svc_name_, ''))
        self.main()

    def main(self):
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        wol_script = os.path.join(script_dir, 'router_wol.py')
        
        # Start the router_wol.py script in a subprocess
        process = subprocess.Popen([sys.executable, wol_script])
        
        # Wait for service stop signal
        while self.is_running:
            rc = win32event.WaitForSingleObject(self.hWaitStop, 5000)
            if rc == win32event.WAIT_OBJECT_0:
                # Stop was signaled, terminate the process
                process.terminate()
                break
        
        # Ensure the process is terminated
        if process.poll() is None:
            process.terminate()
            process.wait()

if __name__ == '__main__':
    if len(sys.argv) == 1:
        servicemanager.Initialize()
        servicemanager.PrepareToHostSingle(WoLService)
        servicemanager.StartServiceCtrlDispatcher()
    else:
        win32serviceutil.HandleCommandLine(WoLService) 