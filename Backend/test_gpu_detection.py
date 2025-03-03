import torch
import os
import sys
from termcolor import colored

def test_gpu_detection():
    """Test GPU detection and configuration"""
    
    print(colored("\n=== Testing GPU Detection ===", "blue"))
    
    # Check Python version
    print(colored(f"Python version: {sys.version}", "cyan"))
    
    # Check PyTorch version
    print(colored(f"PyTorch version: {torch.__version__}", "cyan"))
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(colored(f"CUDA available: {cuda_available}", "green" if cuda_available else "red"))
    
    if cuda_available:
        # Get CUDA version
        cuda_version = torch.version.cuda
        print(colored(f"CUDA version: {cuda_version}", "cyan"))
        
        # Get device count
        device_count = torch.cuda.device_count()
        print(colored(f"GPU device count: {device_count}", "cyan"))
        
        # Get device properties for each GPU
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            device_capability = torch.cuda.get_device_capability(i)
            print(colored(f"GPU {i}: {device_name} (Compute Capability: {device_capability[0]}.{device_capability[1]})", "green"))
            
            # Check if device is initialized
            try:
                torch.cuda.set_device(i)
                current_device = torch.cuda.current_device()
                print(colored(f"  Current device: {current_device}", "cyan"))
                
                # Test memory allocation
                try:
                    test_tensor = torch.zeros(1, device=f'cuda:{i}')
                    print(colored(f"  Memory allocation test: Success", "green"))
                    del test_tensor
                except Exception as e:
                    print(colored(f"  Memory allocation test: Failed - {str(e)}", "red"))
                
            except Exception as e:
                print(colored(f"  Error setting device: {str(e)}", "red"))
    else:
        # Check why CUDA is not available
        if not hasattr(torch, 'cuda'):
            print(colored("PyTorch was not built with CUDA support", "red"))
        else:
            # Check CUDA_HOME environment variable
            cuda_home = os.environ.get('CUDA_HOME')
            print(colored(f"CUDA_HOME environment variable: {cuda_home if cuda_home else 'Not set'}", "yellow"))
            
            # Check PATH for CUDA
            path = os.environ.get('PATH', '')
            cuda_in_path = any('cuda' in p.lower() for p in path.split(os.pathsep))
            print(colored(f"CUDA in PATH: {cuda_in_path}", "yellow"))
            
            # Check for NVIDIA driver
            try:
                import subprocess
                if sys.platform == 'win32':
                    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
                    if result.returncode == 0:
                        print(colored("NVIDIA driver is installed and working", "green"))
                        print(colored("nvidia-smi output:", "cyan"))
                        print(result.stdout[:500])  # Print first 500 chars of output
                    else:
                        print(colored("NVIDIA driver not found or not working", "red"))
                else:
                    # Linux/Mac
                    result = subprocess.run(['which', 'nvidia-smi'], capture_output=True, text=True)
                    if result.returncode == 0:
                        nvidia_smi_path = result.stdout.strip()
                        print(colored(f"nvidia-smi found at: {nvidia_smi_path}", "green"))
                        
                        # Run nvidia-smi
                        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
                        if result.returncode == 0:
                            print(colored("NVIDIA driver is installed and working", "green"))
                            print(colored("nvidia-smi output:", "cyan"))
                            print(result.stdout[:500])  # Print first 500 chars of output
                        else:
                            print(colored("NVIDIA driver not working properly", "red"))
                    else:
                        print(colored("nvidia-smi not found", "red"))
            except Exception as e:
                print(colored(f"Error checking NVIDIA driver: {str(e)}", "red"))
    
    print(colored("\n=== GPU Detection Test Complete ===", "blue"))

if __name__ == "__main__":
    test_gpu_detection() 