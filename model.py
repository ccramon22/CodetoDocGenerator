from transformers import T5ForConditionalGeneration, RobertaTokenizer
from transformers import Trainer, TrainingArguments
from datasets import Dataset
from tqdm.auto import tqdm

def load_model_and_tokenizer(model_name="Salesforce/codet5-base"):
    """Load pretrained CodeT5 model and tokenizer."""
    print(f"Downloading tokenizer for {model_name}...")
    tokenizer = RobertaTokenizer.from_pretrained(model_name, use_fast=True)

    print(f"Downloading model {model_name}... This might take a while for large models.")

    model = T5ForConditionalGeneration.from_pretrained(model_name, use_cache=True,)

    print("Model and tokenizer loaded successfully!")
    return model, tokenizer


def tokenize_and_prepare(examples, tokenizer, max_length=512):
    """Tokenize inputs and targets for model training."""
    model_inputs = tokenizer(
        examples['input'],
        max_length=max_length,
        padding="max_length",
        truncation=True
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples['target'],
            max_length=max_length,
            padding="max_length",
            truncation=True
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def train_documentation_model(df, model_name="Salesforce/codet5-base", output_dir="./codet5_documentation_generator",
                              force_intel=False):
    """Train the documentation generation model.

    Args:
        df: DataFrame containing the training data
        model_name: Pre-trained model to use
        output_dir: Directory to save the trained model
        force_intel: If True, will error if Intel GPU is not available
    """
    import sys
    import torch

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
                print("No GPU found, using CPU")
                use_fp16 = False
                use_bf16 = False

    # Create dataset
    print('Creating Dataset')
    dataset = Dataset.from_pandas(df)

    # Load model and tokenizer
    print('Loading Model and Tokenizer')
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Move model to the appropriate device
    model = model.to(device)

    # Tokenize dataset
    print('Stand back, Tokenizing in progress')
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_and_prepare(examples, tokenizer),
        batched=True
    )

    # Split dataset
    print('Splitting dataset for training and testing')
    train_test_split = tokenized_dataset.train_test_split(test_size=0.1)

    # Set batch size based on device
    batch_size = 4

    # Set up training arguments
    print('Setting up training parameters')
    training_args = TrainingArguments(
        output_dir="./training_results",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=2,
        weight_decay=0.01,
        save_total_limit=3,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_test_split["train"],
        eval_dataset=train_test_split["test"],
    )

    # Train model
    trainer.train()

    # Save model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return model, tokenizer