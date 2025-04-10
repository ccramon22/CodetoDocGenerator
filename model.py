from transformers import T5ForConditionalGeneration, RobertaTokenizer
from transformers import Trainer, TrainingArguments
from datasets import Dataset


def load_model_and_tokenizer(model_name="Salesforce/codet5-large"):
    """Load pretrained CodeT5 model and tokenizer."""
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
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


def train_documentation_model(df, model_name="Salesforce/codet5-large", output_dir="./codet5_documentation_generator"):
    """Train the documentation generation model."""
    # Create dataset
    dataset = Dataset.from_pandas(df)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Tokenize dataset
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_and_prepare(examples, tokenizer),
        batched=True
    )

    # Split dataset
    train_test_split = tokenized_dataset.train_test_split(test_size=0.1)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./training_results",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
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