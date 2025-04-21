from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import torch
import os
from analytics import TrainingAnalytics, AnalyticsCallback
from evaluation import generate_documentation


def load_model_and_tokenizer(model_name="mistralai/Mistral-7B-v0.3"):
    """Load pretrained Mistral model and tokenizer."""
    print(f"Checking if model {model_name} is already downloaded...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        local_files_only=False
    )

    # Set padding token for Mistral
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer loaded successfully and padding token set.")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16
    )
    print("Model loaded successfully!")

    return model, tokenizer


def tokenize_dataset(df, tokenizer, max_length=512):
    """Tokenize dataset for the model."""
    from datasets import Dataset

    dataset = Dataset.from_pandas(df)

    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples['input'],
            max_length=max_length,
            padding="max_length",
            truncation=True
        )

        # For causal LM, labels are input_ids
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs

    return dataset.map(tokenize_function, batched=True)


def train_documentation_model(df, model_name="mistralai/Mistral-7B-v0.3",
                              output_dir="./mistral_documentation_generator",
                              force_intel=False,
                              recovery_mode=False):
    """Train the documentation generation model with optimized memory usage for Intel A770."""
    import sys
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig

    # Create analytics object
    analytics = TrainingAnalytics()

    # Force Intel GPU with optimized settings
    try:
         #####import intel_extension_for_pytorch as ipex
        # if not torch.xpu.is_available():
        #     print("ERROR: Intel GPU extensions are installed but no Intel GPU is available.")
        #     sys.exit(1)
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
        save_total_limit=1,
        load_best_model_at_end=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
        fp16=use_fp16,
        bf16=use_bf16,
        warmup_steps=100,  # Reduced
        max_steps=recovery_mode and 1000 or 2000,  # Reduced further in recovery mode
        report_to="none",
        remove_unused_columns=True,
        optim="adamw_torch_xpu" if hasattr(torch, 'xpu') else "adamw_torch",
        gradient_checkpointing=True,  # Enable gradient checkpointing
        dataloader_num_workers=1,  # Limit workers
        group_by_length=True,  # Group by sequence length for efficiency
    )

    # Initialize trainer with callback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_test_split["train"],
        eval_dataset=train_test_split["test"],
        callbacks=[AnalyticsCallback(analytics)]
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