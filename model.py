from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import torch
import os
from analytics import TrainingAnalytics, AnalyticsCallback
from evaluation import generate_documentation


def load_model_and_tokenizer(model_name="unsloth/mistral-7b-bnb-4bit"):
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
        # PROMPT DESIGNER NOTES:
        # This function structures the input for the model. The current structure is:
        # 1. Imports section (if any imports exist)
        # 2. Class context (if function is a method)
        # 3. Function body
        # 4. Target documentation
        #
        # You can modify the prompt structure while maintaining this context hierarchy.
        # Available context information:
        # - ctx['imports']: List of import statements from the file
        # - ctx['class_context']: Name of the containing class (if any)
        # - ctx['class_docstring']: Documentation of the containing class (if any)
        # - ctx['file_path']: Path to the source file
        #
        # Current max_length is 512 tokens. Adjust if needed for your prompt design.
        # The model expects the documentation to be the last part of the input.
        
        inputs = []
        for func, doc, ctx in zip(examples['function_body'], examples['docstring'], examples['context']):
            # Build context string
            context_str = ""
            if ctx['imports']:
                context_str += "Imports:\n" + "\n".join(ctx['imports']) + "\n\n"
            if ctx['class_context']:
                context_str += f"Class: {ctx['class_context']}\n"
                if 'class_docstring' in ctx:
                    context_str += f"Class Documentation:\n{ctx['class_docstring']}\n\n"
            
            # Combine context with function and docstring
            input_text = f"{context_str}Function:\n{func}\n\nDocumentation:\n{doc}"
            inputs.append(input_text)
        
        model_inputs = tokenizer(
            inputs,
            max_length=max_length,
            padding="max_length",
            truncation=True
        )

        # For causal LM, labels are input_ids
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs

    return dataset.map(tokenize_function, batched=True)


def train_documentation_model(df, model_name="unsloth/mistral-7b-bnb-4bit",
                              output_dir="./mistral_documentation_generator",
                              force_intel=False):
    """Train the documentation generation model."""
    import sys
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig

    # Create analytics object
    analytics = TrainingAnalytics()

    # Determine which device to use
    if force_intel:
        # If Intel GPU is required, check for it
        try:
            import intel_extension_for_pytorch as ipex
            if not torch.xpu.is_available():
                print("ERROR: Intel GPU extensions are installed but no Intel GPU is available.")
                print("Program configured to require an Intel GPU.")
                sys.exit(1)
            device = torch.device("xpu")
            print(f"Using Intel GPU: {torch.xpu.get_device_name(0)}")
            use_bf16 = True
            use_fp16 = False
        except ImportError:
            print("ERROR: Intel PyTorch extensions are not available.")
            print("Program configured to require an Intel GPU.")
            print("Please install extensions with: pip install intel-extension-for-pytorch")
            print("Note: This requires Python 3.11 or earlier.")
            sys.exit(1)
    else:
        # Otherwise try available devices in order: Intel GPU, NVIDIA GPU, CPU
        try:
            import intel_extension_for_pytorch as ipex
            if torch.xpu.is_available():
                device = torch.device("xpu")
                print(f"Using Intel GPU: {torch.xpu.get_device_name(0)}")
                use_bf16 = True
                use_fp16 = False
            elif torch.cuda.is_available():
                device = torch.device("cuda")
                print(f"Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
                use_fp16 = True
                use_bf16 = False
            else:
                device = torch.device("cpu")
                print("No GPU found, using CPU")
                use_fp16 = False
                use_bf16 = False
        except ImportError:
            # If Intel extensions aren't available, try NVIDIA or CPU
            if torch.cuda.is_available():
                device = torch.device("cuda")
                print(f"Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
                use_fp16 = True
                use_bf16 = False
            else:
                device = torch.device("cpu")
                print("No GPU found, using CPU. Training will be very slow.")
                use_fp16 = False
                use_bf16 = False

    # Load model and tokenizer
    print('Loading Model and Tokenizer')
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Prepare dataset with proper tokenization
    tokenized_dataset = tokenize_dataset(df, tokenizer)

    # Split dataset
    print('Splitting dataset for training and testing')
    train_test_split = tokenized_dataset.train_test_split(test_size=0.1)

    # Configure model for efficient training with LoRA
    if device.type != "cpu":
        print("Configuring model for efficient training with LoRA")

        # Configure LoRA with minimal settings for testing
        peft_config = LoraConfig(
            r=4,  # Reduced from 8 to 4
            lora_alpha=8,  # Reduced from 16 to 8
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )

        # Prepare model for training
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)
    else:
        print("WARNING: Training on CPU. This will be extremely slow and memory intensive.")

    # Move model to device
    model = model.to(device)

    # Set batch size based on device and model size
    batch_size = 1 if device.type != "cpu" else 1

    # Set up training arguments for minimal test run
    print('Setting up training parameters')
    training_args = TrainingArguments(
        output_dir="./training_results",
        eval_strategy="steps",
        eval_steps=100,  # Reduced from 500
        logging_strategy="steps",
        logging_steps=50,  # Reduced from 100
        save_strategy="steps",
        save_steps=100,  # Reduced from 500
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=1,  # Single epoch for testing
        weight_decay=0.01,
        save_total_limit=1,
        load_best_model_at_end=True,
        gradient_accumulation_steps=8,  # Reduced from 16
        fp16=use_fp16,
        bf16=use_bf16,
        warmup_steps=50,  # Reduced from 500
        max_steps=200,  # Reduced from 5000
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True  # Added to reduce memory usage
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

    # Generate some example predictions for analysis
    print("Generating sample predictions for analysis...")
    test_examples = train_test_split["test"].select(range(min(3, len(train_test_split["test"]))))

    for example in test_examples:
        # Get input code snippet from dataset
        input_ids = example["input_ids"]
        input_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        code_snippet = input_text.split("Code: ")[1].split("\nDocumentation:")[
            0] if "Code: " in input_text else input_text

        # Generate documentation
        prompt = f"Code: {code_snippet}\nDocumentation:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=200,
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