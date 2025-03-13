import torch
from termcolor import colored

def test_cuda():
    print(colored("\n=== Testing CUDA Availability ===", "cyan"))
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        
        # Test CUDA with a simple tensor operation
        print("\nTesting CUDA with tensor operation...")
        x = torch.rand(5, 3)
        if torch.cuda.is_available():
            x = x.cuda()
            print(colored("✓ Successfully moved tensor to CUDA", "green"))
            print(f"Tensor device: {x.device}")
    else:
        print(colored("✗ CUDA is not available", "red"))
        print("Please check your PyTorch installation and NVIDIA drivers")

if __name__ == "__main__":
    test_cuda() 