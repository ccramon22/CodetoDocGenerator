from data_collection import clone_repositories, extract_python_files
from preprocessing import extract_function_docstring_pairs, prepare_dataset
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

    # Clone repositories
    clone_repositories(repos, repos_dir)

    # Extract Python files
    python_files = extract_python_files(repos_dir)
    print(f"Found {len(python_files)} Python files")

    # Extract function-docstring pairs
    pairs = extract_function_docstring_pairs(python_files)
    print(f"Extracted {len(pairs)} function-docstring pairs")

    # Prepare dataset
    df = prepare_dataset(pairs)

    # Train model
    model, tokenizer = train_documentation_model(df)

    print("Training complete! Model saved to ./codet5_documentation_generator")

    # Test the model with a sample code snippet
    test_snippet = "def calculate_average(numbers):"
    generated_doc = generate_documentation(test_snippet, model, tokenizer)
    print(f"Sample documentation generation:\nInput: {test_snippet}\nGenerated: {generated_doc}")


if __name__ == "__main__":
    main()