from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import torch
import os
from analytics import TrainingAnalytics, AnalyticsCallback
from evaluation import generate_documentation
import gc
import random
import json
from pathlib import Path
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import logging
from datetime import datetime
from torch.optim import AdamW
import shutil
import time
from transformers import BitsAndBytesConfig

# Set up logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(model_name="mistralai/Mistral-7B-v0.3"):
    """Load pretrained Mistral model and tokenizer."""
    print(f"Checking if model {model_name} is already downloaded...")

    # Check available devices
    if not torch.xpu.is_available():
        raise RuntimeError("Intel XPU is not available. Please ensure Intel GPU drivers are installed.")
    
    print("Intel XPU is available")
    print(f"XPU Device: {torch.xpu.get_device_name(0)}")
    print(f"XPU Memory: {torch.xpu.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Clear XPU cache
    torch.xpu.empty_cache()
    gc.collect()
    print("Cleared device caches and memory")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        local_files_only=False
    )

    # Set padding token
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer loaded successfully and padding token set.")

    # Load model with XPU optimization
    print("Loading model with XPU optimization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        use_cache=False,
        use_safetensors=True,
        ignore_mismatched_sizes=True,
        local_files_only=False,
        revision="main",
        trust_remote_code=True
    )
    
    # Move model to XPU
    model = model.to('xpu')
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    print("Model loaded successfully!")
    return model, tokenizer


def tokenize_dataset(df, tokenizer, max_length=512):
    """Tokenize dataset for the model with one-shot learning and chain-of-thought."""
    from datasets import Dataset
    import random
    import os
    import json
    from pathlib import Path

    # Define cache paths
    cache_dir = Path("data/cache")
    cache_dir.mkdir(exist_ok=True)
    tokenized_cache_path = cache_dir / "tokenized_dataset"
    mapping_cache_path = cache_dir / "example_mapping.json"

    # Check if we have cached results
    if tokenized_cache_path.exists() and mapping_cache_path.exists():
        print("Loading cached tokenized dataset...")
        try:
            # Load the cached dataset
            tokenized_dataset = Dataset.load_from_disk(str(tokenized_cache_path))
            
            # Load the example mapping
            with open(mapping_cache_path, 'r') as f:
                example_mapping = json.load(f)
            
            print("Successfully loaded cached dataset!")
            return tokenized_dataset
        except Exception as e:
            print(f"Error loading cached dataset: {e}")
            print("Proceeding with fresh tokenization...")

    dataset = Dataset.from_pandas(df)

    def tokenize_function(examples):
        inputs = []
        labels = []
        example_mapping = {}  # Store which examples were paired together
        
        for idx, (func, doc, ctx) in enumerate(zip(examples['function_body'], examples['docstring'], examples['context'])):
            # Build context string
            context_str = ""
            if ctx and 'imports' in ctx:
                context_str += "Imports:\n" + "\n".join(ctx['imports']) + "\n\n"
            if ctx and 'class_context' in ctx:
                context_str += f"Class: {ctx['class_context']}\n"
                if 'class_docstring' in ctx:
                    context_str += f"Class Documentation:\n{ctx['class_docstring']}\n\n"
            
            # Add one-shot example
            # Get a random example from the dataset that's not the current one
            example_idx = random.randint(0, len(dataset) - 1)
            example = dataset[example_idx]
            while example['function_body'] == func:  # Ensure we don't use the same function
                example_idx = random.randint(0, len(dataset) - 1)
                example = dataset[example_idx]
            
            # Store the mapping
            example_mapping[idx] = example_idx
            
            # Create input text without the target documentation
            input_text = f"""Example Function:
{example['function_body']}

Let's analyze this function step by step:
1. First, I identify the function's purpose and main operations
2. Then, I examine the parameters and their types
3. Next, I look for return values and their types
4. Finally, I consider any important side effects or exceptions

Based on this analysis, here's the documentation:
{example['docstring']}

Now, let's document this new function:
{func}

Let's analyze this function step by step:
1. First, I identify the function's purpose and main operations
2. Then, I examine the parameters and their types
3. Next, I look for return values and their types
4. Finally, I consider any important side effects or exceptions

Based on this analysis, here's the documentation:"""

            # Create full text with documentation for labels
            full_text = input_text + f"\n{doc}"

            # Tokenize input and full text
            tokenized_input = tokenizer(
                input_text,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            tokenized_full = tokenizer(
                full_text,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            # Convert to lists for dataset compatibility
            input_dict = {k: v[0].tolist() for k, v in tokenized_input.items()}
            label_dict = {k: v[0].tolist() for k, v in tokenized_full.items()}
            
            inputs.append(input_dict)
            labels.append(label_dict)
        
        # Combine all tokenized inputs and labels
        combined = {
            "input_ids": [item["input_ids"] for item in inputs],
            "attention_mask": [item["attention_mask"] for item in inputs],
            "labels": [item["input_ids"] for item in labels]  # Use input_ids as labels
        }

        # Save the example mapping
        with open(mapping_cache_path, 'w') as f:
            json.dump(example_mapping, f)
        
        return combined

    # Map the tokenization function to the dataset
    print("Tokenizing dataset (this may take a while)...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=1,  # Process one at a time to handle random example selection
        remove_columns=dataset.column_names  # Remove original columns
    )
    
    # Save the tokenized dataset
    print("Saving tokenized dataset to cache...")
    tokenized_dataset.save_to_disk(str(tokenized_cache_path))
    print("Tokenized dataset saved successfully!")
    
    return tokenized_dataset


def train_documentation_model(df, model_name="mistralai/Mistral-7B-v0.3",
                            output_dir="./mistral_documentation_generator",
                            force_intel=False,
                            recovery_mode=False):
    """Train the documentation generation model with optimized memory usage for Intel A770."""
    import sys
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    import shutil
    import os
    import time
    import logging

    # Close any existing log handlers
    for handler in logging.root.handlers[:]:
        handler.close()
        logging.root.removeHandler(handler)

    # Clear logs directory with retry
    logs_dir = "./logs"
    max_retries = 3
    retry_delay = 1  # seconds

    for attempt in range(max_retries):
        try:
            if os.path.exists(logs_dir):
                print(f"Clearing logs directory: {logs_dir}")
                shutil.rmtree(logs_dir)
            os.makedirs(logs_dir, exist_ok=True)
            break
        except PermissionError:
            if attempt < max_retries - 1:
                print(f"Log directory is in use, waiting {retry_delay} seconds before retry...")
                time.sleep(retry_delay)
            else:
                print("Warning: Could not clear logs directory, continuing with existing logs")
                break

    # Create analytics object
    analytics = TrainingAnalytics()

    # Force Intel GPU with optimized settings
    try:
        device = torch.device("xpu")
        print(f"Using Intel GPU: {torch.xpu.get_device_name(0)}")
        use_bf16 = True
        use_fp16 = False
    except ImportError:
        print("ERROR: Intel PyTorch extensions are not available.")
        sys.exit(1)

    # Adjust based on recovery mode
    if recovery_mode:
        print("Running in recovery mode with aggressive memory optimization")
        load_in_4bit = True
        load_in_8bit = False
        max_length = 256  # Further reduced
        gradient_accumulation_steps = 64  # More aggressive
        batch_size = 1
        lora_rank = 4  # Smaller LoRA rank
    else:
        load_in_4bit = False
        load_in_8bit = True
        max_length = 384
        gradient_accumulation_steps = 32
        batch_size = 1
        lora_rank = 8

    # Add quantization configuration
    print(f"Loading Model and Tokenizer with {'4-bit' if load_in_4bit else '8-bit'} quantization")

    # Configure quantization
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_quant_type="nf4"
        )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer loaded successfully and padding token set.")

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",  # Better memory management
        offload_folder="offload",  # Enable CPU offloading if needed
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16
    )
    print(f"Model loaded successfully with {4 if load_in_4bit else 8}-bit quantization!")

    # Prepare dataset with proper tokenization
    print(f"Tokenizing dataset with sequence length {max_length}")
    tokenized_dataset = tokenize_dataset(df, tokenizer, max_length=max_length)

    # Free up memory
    import gc
    gc.collect()
    torch.xpu.empty_cache() if hasattr(torch, 'xpu') else torch.cuda.empty_cache()

    # Split dataset
    print('Splitting dataset for training and testing')
    train_test_split = tokenized_dataset.train_test_split(test_size=0.05)  # Small test size

    # Configure model for efficient training with LoRA
    print(f"Configuring model for efficient training with LoRA (rank={lora_rank})")

    # LoRA config with memory optimization
    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    # Print trainable parameters vs total
    model.print_trainable_parameters()

    # Set up training arguments with memory optimizations
    print('Setting up training parameters')
    training_args = TrainingArguments(
        output_dir="./training_results",
        eval_strategy="steps",
        eval_steps=1000,  # Less frequent evaluation
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="steps",
        save_steps=1000,  # Less frequent saving
        learning_rate=2e-4,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=3,  # Reduced from 5
        weight_decay=0.01,
        warmup_steps=100,  # Reduced
        max_steps=recovery_mode and 1000 or 2000,  # Reduced further in recovery mode
        report_to="none",
        remove_unused_columns=True,
        optim="adamw_torch_xpu" if hasattr(torch, 'xpu') else "adamw_torch",
        gradient_checkpointing=True,  # Enable gradient checkpointing
        dataloader_num_workers=1,  # Limit workers
        group_by_length=True,  # Group by sequence length for efficiency
        gradient_accumulation_steps=gradient_accumulation_steps,
        fp16=use_fp16,
        bf16=use_bf16
    )

    # Train model
    print("Starting training...")
    trainer.train()

    # End training analytics
    analytics.end_training()

    # Save model
    print("Saving model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Generate fewer example predictions for analysis
    print("Generating sample predictions for analysis...")
    test_examples = train_test_split["test"].select(range(min(2, len(train_test_split["test"]))))

    for example in test_examples:
        # Get input code snippet from dataset
        input_ids = example["input_ids"]
        input_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        code_snippet = input_text.split("Code: ")[1].split("\nDocumentation:")[
            0] if "Code: " in input_text else input_text

        # Generate documentation with reduced tokens
        prompt = f"Code: {code_snippet}\nDocumentation:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=100,  # Reduced from 200
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_doc = generated_text.split("Documentation:")[
            1].strip() if "Documentation:" in generated_text else generated_text

        print("\nCode snippet:", code_snippet[:100] + "..." if len(code_snippet) > 100 else code_snippet)
        print("\nGenerated documentation:", generated_doc)

    # Create analytics dashboard
    print("Creating analytics dashboard...")
    print("Analytics dashboard created in ./analytics directory")

    return model, tokenizer, analytics