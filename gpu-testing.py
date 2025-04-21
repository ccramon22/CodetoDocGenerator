# gpu_test.py
import torch
import time
import os


def test_gpu():
    print("Testing GPU availability for PyTorch...")

    # Test CUDA (NVIDIA)
    print("\n=== NVIDIA GPU Test ===")
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

        # Run a simple CUDA operation
        a = torch.randn(10000, 10000).cuda()
        b = torch.randn(10000, 10000).cuda()

        start = time.time()
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        end = time.time()

        print(f"CUDA matrix multiplication time: {end - start:.4f} seconds")
    else:
        print("CUDA not available")

    # Test XPU (Intel)
    print("\n=== Intel GPU Test ===")
    try:
        #import intel_extension_for_pytorch as ipex
        #print("Intel PyTorch extension installed")

        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            print(f"XPU available: {torch.xpu.is_available()}")
            print(f"XPU device count: {torch.xpu.device_count()}")
            print(f"XPU device name: {torch.xpu.get_device_name(0)}")

            # Run a simple XPU operation
            a = torch.randn(10000, 10000).xpu()
            b = torch.randn(10000, 10000).xpu()

            start = time.time()
            c = torch.matmul(a, b)
            torch.xpu.synchronize()
            end = time.time()

            print(f"XPU matrix multiplication time: {end - start:.4f} seconds")
        else:
            print("XPU not available")
    except ImportError:
        print("Intel PyTorch extension not installed")
    except Exception as e:
        print(f"Error testing Intel GPU: {e}")

    # Test CPU as benchmark
    print("\n=== CPU Test ===")
    a = torch.randn(5000, 5000)  # Smaller size for CPU
    b = torch.randn(5000, 5000)

    start = time.time()
    c = torch.matmul(a, b)
    end = time.time()

    print(f"CPU matrix multiplication time: {end - start:.4f} seconds")

    # Test Mistral model loading
    print("\n=== Testing Mistral Model Loading ===")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("Loading Mistral tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")
        tokenizer.pad_token = tokenizer.eos_token
        print("Tokenizer loaded successfully.")

        print("Attempting to load a small portion of Mistral model...")
        model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-v0.3",
            torch_dtype=torch.float16,
            device_map="auto",  # Let transformers decide best device
            max_memory={0: "4GB"}  # Limit memory usage
        )
        print(f"Model loaded on device: {next(model.parameters()).device}")

        # Run a simple inference test
        inputs = tokenizer("def hello_world():", return_tensors="pt").to(model.device)

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(inputs["input_ids"], max_new_tokens=20)
        end = time.time()

        print(f"Generation time: {end - start:.4f} seconds")
        print(f"Generated: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")

    except Exception as e:
        print(f"Error testing model loading: {e}")


if __name__ == "__main__":
    test_gpu()