from transformers import T5ForConditionalGeneration, RobertaTokenizer
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import torch
import os
from analytics import TrainingAnalytics, AnalyticsCallback
from evaluation import generate_documentation


def load_model_and_tokenizer(model_name="Salesforce/codet5-base"):
    """Load pretrained CodeT5 model and tokenizer."""
    print(f"Checking if model {model_name} is already downloaded...")

    # Check if tokenizer is already downloaded
    tokenizer = RobertaTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        local_files_only=False  # Will try local first, then download if needed
    )
    print(f"Tokenizer loaded successfully.")

    # Check if model is already downloaded
    try:
        print(f"Attempting to load model from cache...")
        model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            local_files_only=True  # Only use local files
        )
        print("Model loaded from cache successfully!")
    except Exception as e:
        print(f"Model not found in cache. Downloading model {model_name}...")
        model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            use_cache=True
        )
        print("Model downloaded and loaded successfully!")

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


def train_documentation_model(df, model_name="Salesforce/codet5-base", output_dir="./codet5_documentation_generator"):
    """Train the documentation generation model."""
    # Create analytics object
    analytics = TrainingAnalytics()

    # Create dataset
    print('Creating Dataset')
    dataset = Dataset.from_pandas(df)

    # Load model and tokenizer
    print('Loading Model and Tokenizer')
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Force CPU usage to avoid GPU issues
    device = torch.device("cpu")
    print("Using CPU for training")

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

    # Set up training arguments
    print('Setting up training parameters')
    training_args = TrainingArguments(
        output_dir="./training_results",
        evaluation_strategy="epoch",
        learning_rate=7e-5,
        per_device_train_batch_size=16,  # Reasonable size for CPU
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,
        num_train_epochs=5,  # Reduce epochs for faster testing
        weight_decay=0.01,
        save_total_limit=3,
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
    trainer.train()

    # End training analytics
    analytics.end_training()

    # Save model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Generate some example predictions for analysis
    print("Generating sample predictions for analysis...")
    test_examples = train_test_split["test"].select(range(min(10, len(train_test_split["test"]))))

    for example in test_examples:
        code_snippet = example["input"]
        generated_doc = generate_documentation(code_snippet, model, tokenizer)
        # The new analytics doesn't have a log_prediction method, but we could add one
        # or just print the predictions for now
        print("\nCode snippet:", code_snippet[:100] + "..." if len(code_snippet) > 100 else code_snippet)
        print("\nGenerated documentation:", generated_doc)

    # Create analytics dashboard for presentation
    print("Creating analytics dashboard...")
    # This is automatically done in end_training()

    print("Analytics dashboard created in ./analytics directory")

    return model, tokenizer, analytics