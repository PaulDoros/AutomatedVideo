import schedule
import time
import datetime
import subprocess
import socket
import struct

def wake_pc(mac_address):
    """
    Send a Wake-on-LAN packet to the specified MAC address
    """
    # Parse MAC address
    mac_bytes = bytes.fromhex(mac_address.replace(':', '').replace('-', ''))
    
    # Create magic packet (6 bytes of 0xFF followed by 16 repetitions of the MAC address)
    magic_packet = b'\xff' * 6 + mac_bytes * 16
    
    # Send packet using UDP broadcast
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        s.sendto(magic_packet, ('255.255.255.255', 9))
    
    print(f"[{datetime.datetime.now()}] Wake-on-LAN packet sent to {mac_address}")

# Your PC's MAC address (replace with the one you tested with)
MAC_ADDRESS = "70-D8-23-5C-3F-64"  # Wi-Fi adapter MAC address

# Schedule wake-up times (30 minutes before each scheduled video time)
# Based on your scheduler.py and CHANNEL_CONFIG

# Tech Humor (main) - 14:00 and 18:00
schedule.every().day.at("13:30").do(wake_pc, MAC_ADDRESS)
schedule.every().day.at("17:30").do(wake_pc, MAC_ADDRESS)

# Business - 10:00 and 16:00
schedule.every().day.at("09:30").do(wake_pc, MAC_ADDRESS)
schedule.every().day.at("15:30").do(wake_pc, MAC_ADDRESS)

# Gaming - 20:00
schedule.every().day.at("19:30").do(wake_pc, MAC_ADDRESS)

# Tech - 15:00
schedule.every().day.at("14:30").do(wake_pc, MAC_ADDRESS)

# Daily schedule generation - 00:30 (based on shorts_scheduler.py)
schedule.every().day.at("00:15").do(wake_pc, MAC_ADDRESS)

# Performance analysis - 02:30 (based on shorts_scheduler.py)
schedule.every().day.at("01:45").do(wake_pc, MAC_ADDRESS)

print(f"[{datetime.datetime.now()}] Wake-on-LAN scheduler started")
print(f"Scheduled wake-up times (30 minutes before each content generation):")
print("- 09:30 (Business channel content at 10:00)")
print("- 13:30 (Tech Humor channel content at 14:00)")
print("- 14:30 (Tech channel content at 15:00)")
print("- 15:30 (Business channel content at 16:00)")
print("- 17:30 (Tech Humor channel content at 18:00)")
print("- 19:30 (Gaming channel content at 20:00)")
print("- 00:15 (Daily schedule generation at 00:30)")
print("- 01:45 (Performance analysis at 02:00)")

# Run the scheduler continuously
while True:
    schedule.run_pending()
    time.sleep(60)  # Check every minute 