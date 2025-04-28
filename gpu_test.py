import torch
import sys
import os
import platform
import subprocess
from pathlib import Path

def check_gpu_availability():
    """Check GPU availability and capabilities."""
    print("=== GPU Availability Check ===")
    
    # Check NVIDIA drivers first
    try:
        nvidia_smi = subprocess.check_output(['nvidia-smi'])
        print("\nNVIDIA Drivers Information:")
        print(nvidia_smi.decode())
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\nNVIDIA drivers not found or nvidia-smi not in PATH")
    
    # Check CUDA (NVIDIA) availability
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print("\nNVIDIA GPU (CUDA) Information:")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            print(f"    Compute Capability: {torch.cuda.get_device_capability(i)}")
    else:
        print("\nNo NVIDIA GPU (CUDA) detected")
        print("Make sure you have:")
        print("1. NVIDIA drivers installed")
        print("2. CUDA Toolkit installed")
        print("3. PyTorch with CUDA support installed")
        print("\nTo install PyTorch with CUDA support:")
        print("pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("\nTo verify NVIDIA drivers:")
        print("1. Open Device Manager")
        print("2. Check under 'Display adapters' for your NVIDIA GPU")
        print("3. Right-click and select 'Properties' to check driver version")
    
    # Check XPU (Intel) availability
    xpu_available = hasattr(torch, 'xpu') and torch.xpu.is_available()
    if xpu_available:
        print("\nIntel GPU (XPU) Information:")
        print(f"Number of XPU devices: {torch.xpu.device_count()}")
        for i in range(torch.xpu.device_count()):
            print(f"  Device {i}: {torch.xpu.get_device_name(i)}")
            print(f"    Memory: {torch.xpu.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    else:
        print("\nNo Intel GPU (XPU) detected")
        print("Make sure you have:")
        print("1. Intel GPU drivers installed")
        print("2. Intel oneAPI Base Toolkit installed")
        print("3. intel-extension-for-pytorch installed")
        print("You can install Intel Extension for PyTorch using:")
        print("pip install intel-extension-for-pytorch")
    
    # Check CPU information
    print("\nCPU Information:")
    print(f"Number of CPU cores: {os.cpu_count()}")
    print(f"CPU architecture: {platform.machine()}")
    print(f"System: {platform.system()} {platform.release()}")
    
    # Check PyTorch version and build
    print("\nPyTorch Information:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch build with CUDA: {torch.version.cuda is not None}")
    print(f"PyTorch build with XPU: {hasattr(torch, 'xpu')}")
    
    # Save results to file
    results = {
        "cuda_available": cuda_available,
        "xpu_available": xpu_available,
        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if cuda_available else [],
        "xpu_devices": [torch.xpu.get_device_name(i) for i in range(torch.xpu.device_count())] if xpu_available else [],
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.version.cuda else None,
        "system_info": {
            "cpu_cores": os.cpu_count(),
            "architecture": platform.machine(),
            "system": platform.system(),
            "release": platform.release()
        }
    }
    
    # Create results directory if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Save results to JSON file
    import json
    with open(results_dir / "gpu_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to results/gpu_test_results.json")

if __name__ == "__main__":
    check_gpu_availability() 