import os
import json
import pandas as pd
import torch
import gc
from data_collection import clone_repositories, extract_python_files
from preprocessing import extract_function_docstring_pairs_ast, prepare_dataset
from model import train_documentation_model
from evaluation import evaluate_model, generate_documentation


def main():
    """Main function with optimized memory management for Intel A770 GPU."""
    print("Starting code documentation generator with Intel A770 optimization")

    # Memory management - initial cleanup
    gc.collect()
    if hasattr(torch, 'xpu'):
        torch.xpu.empty_cache()

    # Keep all repositories as requested
    repos = [
        "https://github.com/getsentry/sentry.git",
        "https://github.com/scikit-learn/scikit-learn.git",
        "https://github.com/paperless-ngx/paperless-ngx.git",
        "https://github.com/numpy/numpy.git",
        "https://github.com/django/django.git",
        "https://github.com/fastapi/fastapi.git",
        "https://github.com/matplotlib/matplotlib.git",
        "https://github.com/pandas-dev/pandas.git",
        "https://github.com/pytorch/pytorch.git",
    ]

    # Set directories and paths
    repos_dir = "./github_repos"
    preprocessed_data_path = "data/processed_dataset.json"
    os.makedirs("data", exist_ok=True)

    # Data preparation with improved memory handling
    if os.path.exists(preprocessed_data_path):
        print(f"Loading preprocessed data from {preprocessed_data_path}")
        try:
            # Stream read the JSON file to manage memory
            with open(preprocessed_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            df = pd.DataFrame(data)
            print(f"Loaded dataset with {len(df)} function-docstring pairs")
        except Exception as e:
            print(f"Error loading preprocessed data: {e}")
            print("Will collect new data...")
            df = None
    else:
        df = None

    # If no data loaded, collect new data
    if df is None:
        print("Processing repositories...")

        # Clone repositories
        clone_repositories(repos, repos_dir)

        # Extract Python files
        print("Extracting Python files...")
        python_files = extract_python_files(repos_dir)
        print(f"Found {len(python_files)} Python files")

        # Process in chunks to manage memory
        chunk_size = 200  # Larger chunks but still memory-friendly
        all_pairs = []

        for i in range(0, len(python_files), chunk_size):
            chunk = python_files[i:i + chunk_size]
            print(f"Processing files {i + 1} to {min(i + chunk_size, len(python_files))}")
            pairs = extract_function_docstring_pairs_ast(chunk)
            all_pairs.extend(pairs)

            # Free memory after each chunk
            gc.collect()
            if hasattr(torch, 'xpu'):
                torch.xpu.empty_cache()

        print(f"Extracted {len(all_pairs)} function-docstring pairs")

        # Prepare dataset - using all samples as requested
        df = prepare_dataset(all_pairs)
        print(f"Prepared dataset with {len(df)} samples")

        # Save for future use
        try:
            # Save only essential columns to reduce file size
            save_df = df[['function_name', 'params', 'docstring', 'function_body', 'input', 'output']]
            save_data = save_df.to_dict('records')

            with open(preprocessed_data_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f)
            print(f"Saved preprocessed data to {preprocessed_data_path}")
        except Exception as e:
            print(f"Warning: Could not save preprocessed data: {e}")

    # Force memory cleanup before training
    print("Cleaning memory before training...")
    # Properly handle variable deletion
    if 'python_files' in locals():
        del python_files
    if 'all_pairs' in locals():
        del all_pairs
    if 'pairs' in locals():
        del pairs
    gc.collect()
    if hasattr(torch, 'xpu'):
        torch.xpu.empty_cache()

    # Train model with Intel GPU optimization and all samples
    print("Starting model training with all samples...")
    try:
        model, tokenizer, analytics = train_documentation_model(
            df,
            model_name="mistralai/Mistral-7B-v0.3",
            output_dir="./mistral_documentation_generator",
            force_intel=False  # Force Intel GPU optimization
        )

        print("Training complete! Model saved to ./mistral_documentation_generator")

        # Print analytics information
        print(
            f"Training took {analytics.get_training_time()} with an average of "
            f"{analytics.get_average_seconds_per_step():.2f} seconds per iteration"
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n" + "!" * 80)
            print("OUT OF MEMORY ERROR DETECTED")
            print("The model training exceeded available GPU memory.")
            print("Attempting to recover by reducing batch size and enabling more aggressive memory optimization...")
            print("!" * 80 + "\n")

            # Retry with more aggressive memory settings
            gc.collect()
            if hasattr(torch, 'xpu'):
                torch.xpu.empty_cache()

            # Call with recovery settings
            try:
                model, tokenizer, analytics = train_documentation_model(
                    df,
                    model_name="mistralai/Mistral-7B-v0.3",
                    output_dir="./mistral_documentation_generator",
                    force_intel=True,
                    recovery_mode=True  # Enable more aggressive memory optimization
                )
                print("Recovery successful! Training completed with reduced parameters.")
            except Exception as e2:
                print(f"Recovery attempt failed: {e2}")
                print("Consider reducing the dataset size or using a more efficient model.")
                return
        else:
            print(f"Error during training: {e}")
            return
    except Exception as e:
        print(f"Error during training: {e}")
        return

    # Memory cleanup before evaluation
    gc.collect()
    if hasattr(torch, 'xpu'):
        torch.xpu.empty_cache()

    # Test with sample code snippets
    print("Evaluating model with test cases...")

    test_case1 = "def process_x(data):\n    # Convert data to lowercase\n    data = data.lower()\n    # Remove special characters\n    data = re.sub(r'[^a-z0-9\\s]', '', data)\n    # Remove extra whitespace\n    data = ' '.join(data.split())\n    return data"

    test_case2 = "def func1(df, col_name):\n    # Calculate the mean of the column\n    mean_val = df[col_name].mean()\n    # Replace NaN values with the mean\n    df[col_name].fillna(mean_val, inplace=True)\n    return df"

    test_case3 = "def x(n, k):\n    # Implementation of binary search algorithm\n    low = 0\n    high = n - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if mid == k:\n            return True\n        elif mid < k:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return False"

    test_snippets = [test_case1, test_case2, test_case3]

    try:
        results = evaluate_model(test_snippets, model, tokenizer)

        print("Evaluation Results\n" + "=" * 40)
        for i, result in enumerate(results, 1):
            print(f"\nTest Case {i}:\n")
            print(f"CODE:\n{result['code_snippet']}\n")
            print(f"GENERATED DOCUMENTATION:\n{result['generated_documentation']}\n")
            print("-" * 70)
    except Exception as e:
        print(f"Error during evaluation: {e}")

    print("Process completed.")


if __name__ == "__main__":
    main()