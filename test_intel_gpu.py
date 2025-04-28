import torch
import time
import psutil
import os
import gc
import sys

def print_system_memory():
    """Print system memory usage."""
    process = psutil.Process(os.getpid())
    print(f"System Memory Usage: {process.memory_info().rss / 1024**3:.2f} GB")
    print(f"System Memory Available: {psutil.virtual_memory().available / 1024**3:.2f} GB")

def print_xpu_memory():
    """Print detailed XPU memory information."""
    if torch.xpu.is_available():
        print("\nXPU Memory Information:")
        print(f"Device Count: {torch.xpu.device_count()}")
        print(f"Current Device: {torch.xpu.current_device()}")
        print(f"Device Name: {torch.xpu.get_device_name()}")
        print(f"Memory Allocated: {torch.xpu.memory_allocated() / 1024**3:.2f} GB")
        print(f"Memory Reserved: {torch.xpu.memory_reserved() / 1024**3:.2f} GB")
        print(f"Max Memory Allocated: {torch.xpu.max_memory_allocated() / 1024**3:.2f} GB")
        print(f"Max Memory Reserved: {torch.xpu.max_memory_reserved() / 1024**3:.2f} GB")
        print(f"Memory Stats: {torch.xpu.memory_stats()}")
    else:
        print("XPU is not available")

def test_memory_allocation():
    """Test memory allocation on XPU."""
    if not torch.xpu.is_available():
        print("XPU is not available")
        return

    print("\nTesting Memory Allocation:")
    
    # Clear cache and memory
    torch.xpu.empty_cache()
    gc.collect()
    
    # Initialize list to hold tensors
    tensors = []
    
    # Try incremental allocation
    total_size = 0
    chunk_size = 1024 * 1024 * 1024  # 1GB chunks
    max_size = 14 * 1024 * 1024 * 1024  # 14GB target
    
    try:
        while total_size < max_size:
            print(f"\nCurrent allocation: {total_size / 1024**3:.2f} GB")
            tensor = torch.empty(chunk_size // 4, dtype=torch.float32, device='xpu')
            tensors.append(tensor)  # Keep reference to prevent garbage collection
            total_size += chunk_size
            print(f"Successfully allocated additional 1GB")
            
            # Print current memory stats
            print(f"Memory Allocated: {torch.xpu.memory_allocated() / 1024**3:.2f} GB")
            print(f"Memory Reserved: {torch.xpu.memory_reserved() / 1024**3:.2f} GB")
            
    except Exception as e:
        print(f"Failed to allocate more memory at {total_size / 1024**3:.2f} GB: {str(e)}")
    
    # Clean up
    print("\nCleaning up allocated memory...")
    for tensor in tensors:
        del tensor
    tensors.clear()
    torch.xpu.empty_cache()
    gc.collect()
    
    # Print final memory stats
    print("\nFinal Memory Stats:")
    print(f"Memory Allocated: {torch.xpu.memory_allocated() / 1024**3:.2f} GB")
    print(f"Memory Reserved: {torch.xpu.memory_reserved() / 1024**3:.2f} GB")

def test_model_loading():
    """Test loading a small model to verify XPU functionality."""
    if not torch.xpu.is_available():
        print("XPU is not available")
        return

    print("\nTesting Model Loading:")
    try:
        # Try loading a small model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_name = "facebook/opt-125m"  # Small model for testing
        
        print(f"Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("Model loaded successfully")
        
        # Test inference
        print("Testing inference...")
        inputs = tokenizer("Hello, how are you?", return_tensors="pt").to('xpu')
        outputs = model.generate(**inputs, max_new_tokens=10)
        print(f"Generated text: {tokenizer.decode(outputs[0])}")
        
    except Exception as e:
        print(f"Error during model loading: {str(e)}")

def main():
    print("Starting Intel GPU Tests")
    print("=" * 50)
    
    # Print system information
    print("\nSystem Information:")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Python Version: {sys.version}")
    
    # Print memory information
    print_system_memory()
    print_xpu_memory()
    
    # Run tests
    test_memory_allocation()
    test_model_loading()
    
    print("\nTests completed")

if __name__ == "__main__":
    main() 