# This script automatically enters the PIN code after waking from sleep
# IMPORTANT: This is for automation purposes only and reduces security

# Wait a moment for the lock screen to fully appear
Start-Sleep -Seconds 3

# Send PIN code (3775) with pauses between digits
Add-Type -AssemblyName System.Windows.Forms
[System.Windows.Forms.SendKeys]::SendWait("3")
Start-Sleep -Milliseconds 200
[System.Windows.Forms.SendKeys]::SendWait("7")
Start-Sleep -Milliseconds 200
[System.Windows.Forms.SendKeys]::SendWait("7")
Start-Sleep -Milliseconds 200
[System.Windows.Forms.SendKeys]::SendWait("5")
Start-Sleep -Milliseconds 200

# Press Enter to submit
[System.Windows.Forms.SendKeys]::SendWait("{ENTER}")

Write-Host "PIN code entered automatically" 