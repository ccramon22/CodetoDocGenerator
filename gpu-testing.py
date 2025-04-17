import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version used by PyTorch: {torch.version.cuda}") # This might differ from the driver's max version
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("PyTorch cannot find CUDA. Check installation.")

# Also check accelerate, though this is secondary to PyTorch finding CUDA
try:
    import accelerate
    print(f"Accelerate version: {accelerate.__version__}")
except ImportError:
    print("Accelerate library is not installed.")