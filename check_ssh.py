import socket

def check_port(host, port, timeout=2):
    """Check if a port is open on a host"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    result = sock.connect_ex((host, port))
    sock.close()
    return result == 0

# Check if SSH port is open on the router
router_ip = "192.168.0.1"
ssh_port = 22

print(f"Checking if SSH (port 22) is open on {router_ip}...")
if check_port(router_ip, ssh_port):
    print("✅ SSH port is OPEN. Your router may support SSH access.")
else:
    print("❌ SSH port is CLOSED. Your router probably doesn't have SSH enabled.")
    print("You might need to enable it in the router's web interface if supported.")

# Check common web admin ports too
for port in [80, 443, 8080]:
    if check_port(router_ip, port):
        print(f"Port {port} is open (web interface).") 