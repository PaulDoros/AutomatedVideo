import os
import sys
import subprocess

def run_test():
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set up the environment
    env = os.environ.copy()
    env["PYTHONPATH"] = current_dir
    
    # Run the test_pipeline.py script with the --generate flag
    cmd = [sys.executable, "Backend/test_pipeline.py", "--channel", "tech", "--generate"]
    
    # Run the command
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Print the output in real-time
    for line in process.stdout:
        print(line, end="")
    
    # Wait for the process to complete
    process.wait()
    
    return process.returncode

if __name__ == "__main__":
    sys.exit(run_test()) 