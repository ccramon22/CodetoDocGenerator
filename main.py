import os
import json
import pandas as pd
from data_collection import clone_repositories, extract_python_files
from preprocessing import extract_function_docstring_pairs_ast, prepare_dataset
from model import train_documentation_model
from evaluation import evaluate_model, generate_documentation


def main():
    # Define repositories to clone
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

        # Add more repositories as needed
    ]

    # Set directory for repositories
    repos_dir = "./github_repos"
    preprocessed_data_path = "data/processed_dataset.json"

    # Check if preprocessed data exists
    if os.path.exists(preprocessed_data_path):
        print(f"Loading preprocessed data from {preprocessed_data_path}")
        with open(preprocessed_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    else:
        # Clone repositories
        clone_repositories(repos, repos_dir)

        # Extract Python files
        python_files = extract_python_files(repos_dir)
        print(f"Found {len(python_files)} Python files")

        # Extract function-docstring pairs
        pairs = extract_function_docstring_pairs_ast(python_files)
        print(f"Extracted {len(pairs)} function-docstring pairs")

        # Prepare dataset and save to JSON
        df = prepare_dataset(pairs)

    print(f"Loaded dataset with {len(df)} function-docstring pairs")

    model, tokenizer, analytics = train_documentation_model(df)

    print("Training complete! Model saved to ./codet5_documentation_generator")

    # Print analytics information
    print(
        f"Training took {analytics.get_training_time()} with an average of {analytics.get_average_seconds_per_step():.2f} seconds per iteration")

    # Test the model with a sample code snippet
    test_case1 = "def process_x(data):\n    # Convert data to lowercase\n    data = data.lower()\n    # Remove special characters\n    data = re.sub(r'[^a-z0-9\\s]', '', data)\n    # Remove extra whitespace\n    data = ' '.join(data.split())\n    return data"

    test_case2 = "def func1(df, col_name):\n    # Calculate the mean of the column\n    mean_val = df[col_name].mean()\n    # Replace NaN values with the mean\n    df[col_name].fillna(mean_val, inplace=True)\n    return df"

    test_case3 = "def x(n, k):\n    # Implementation of binary search algorithm\n    low = 0\n    high = n - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if mid == k:\n            return True\n        elif mid < k:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return False"

    test_snippets = [test_case1, test_case2, test_case3]

    results = evaluate_model(test_snippets, model, tokenizer)

    print("Evaluation Results\n" + "=" * 40)
    for i, result in enumerate(results, 1):
        print(f"\nTest Case {i}:\n")
        print(f"CODE:\n{result['code_snippet']}\n")
        print(f"GENERATED DOCUMENTATION:\n{result['generated_documentation']}\n")
        print("-" * 70)


if __name__ == "__main__":
    main()