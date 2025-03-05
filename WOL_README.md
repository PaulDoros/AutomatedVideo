# YouTube Shorts Wake-on-LAN Automation

This system automatically wakes your PC before scheduled YouTube Shorts tasks and puts it back to sleep afterward to save energy.

## Files

- `router_wol.py` - Python script that sends WoL packets to wake the PC
- `sleep_pc.bat` - Batch file that puts the PC to sleep after tasks
- `test_wol.bat` - Test script for Wake-on-LAN functionality
- `setup_tasks.bat` - Creates all the scheduled tasks in Windows Task Scheduler

## Schedule

The system is configured for the following schedule:

| Channel/Task | Wake-up Time | Content Time | Sleep Time |
|--------------|--------------|--------------|------------|
| Business     | 09:30        | 10:00        | 10:30      |
| Tech Humor 1 | 13:30        | 14:00        | 14:30      |
| Tech         | 14:30        | 15:00        | 15:30      |
| Business 2   | 15:30        | 16:00        | 16:30      |
| Tech Humor 2 | 17:30        | 18:00        | 18:30      |
| Gaming       | 19:30        | 20:00        | 20:30      |
| Schedule Gen | 00:15        | 00:30        | 01:15      |
| Analysis     | 01:45        | 02:00        | 02:45      |

## Setup Instructions

1. Run `setup_tasks.bat` as Administrator to set up all scheduled tasks
2. Test the WoL functionality by running `test_wol.bat`
3. Put your PC to sleep and wait for the next scheduled task

## Customizing

- Adjust sleep times in `sleep_pc.bat` if your tasks take longer
- Modify task times in Task Scheduler if you want to change the schedule

## Troubleshooting

If the PC is not waking up:
1. Check Wake-on-LAN is enabled in BIOS/UEFI
2. Verify Device Manager settings for your network adapter
3. Run `test_wol.bat` while your PC is awake to test functionality

## MAC Address

Your PC's Wi-Fi MAC address: `70-D8-23-5C-3F-64` 