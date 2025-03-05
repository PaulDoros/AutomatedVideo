import schedule
import time
import datetime
import socket
import struct
import argparse
import os

def wake_pc(mac_address, broadcast_ip='255.255.255.255'):
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
        s.sendto(magic_packet, (broadcast_ip, 9))
    
    print(f"[{datetime.datetime.now()}] Wake-on-LAN packet sent to {mac_address}")
    
    # Log to file
    try:
        with open('wol_log.txt', 'a') as f:
            f.write(f"{datetime.datetime.now()} - WoL packet sent to {mac_address}\n")
    except Exception as e:
        print(f"Could not write to log file: {e}")

def setup_scheduled_tasks(mac_address):
    """Set up scheduled tasks for YouTube Shorts automation"""
    # Tech Humor (main) - 14:00 and 18:00
    schedule.every().day.at("13:30").do(wake_pc, mac_address)
    schedule.every().day.at("17:30").do(wake_pc, mac_address)

    # Business - 10:00 and 16:00
    schedule.every().day.at("09:30").do(wake_pc, mac_address)
    schedule.every().day.at("15:30").do(wake_pc, mac_address)

    # Gaming - 20:00
    schedule.every().day.at("19:30").do(wake_pc, mac_address)

    # Tech - 15:00
    schedule.every().day.at("14:30").do(wake_pc, mac_address)

    # Daily schedule generation - 00:30
    schedule.every().day.at("00:15").do(wake_pc, mac_address)

    # Performance analysis - 02:30
    schedule.every().day.at("01:45").do(wake_pc, mac_address)

def run_scheduler(mac_address):
    """Run the scheduler continuously"""
    setup_scheduled_tasks(mac_address)
    
    print(f"[{datetime.datetime.now()}] Wake-on-LAN scheduler started")
    print(f"Scheduled wake-up times for MAC: {mac_address}")
    print("- 09:30 (Business channel content at 10:00)")
    print("- 13:30 (Tech Humor channel content at 14:00)")
    print("- 14:30 (Tech channel content at 15:00)")
    print("- 15:30 (Business channel content at 16:00)")
    print("- 17:30 (Tech Humor channel content at 18:00)")
    print("- 19:30 (Gaming channel content at 20:00)")
    print("- 00:15 (Daily schedule generation at 00:30)")
    print("- 01:45 (Performance analysis at 02:00)")
    
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

def send_immediate_wol(mac_address):
    """Send an immediate WoL packet for testing"""
    print(f"Sending immediate Wake-on-LAN packet to {mac_address}")
    wake_pc(mac_address)
    print("Done! If your PC was in sleep mode, it should wake up now.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Wake-on-LAN Scheduler for YouTube Shorts Automation')
    parser.add_argument('--mac', default="70-D8-23-5C-3F-64", 
                        help='MAC address of the PC to wake (default: 70-D8-23-5C-3F-64)')
    parser.add_argument('--test', action='store_true', 
                        help='Send an immediate WoL packet and exit (for testing)')
    parser.add_argument('--broadcast', default='255.255.255.255',
                        help='Broadcast IP address (default: 255.255.255.255)')
    
    args = parser.parse_args()
    
    if args.test:
        send_immediate_wol(args.mac)
    else:
        run_scheduler(args.mac) 