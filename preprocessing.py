import re


def extract_function_docstring_pairs(python_files):
    """Extract function-docstring pairs from Python files."""
    pairs = []

    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

                # Simple regex pattern to extract functions with docstrings
                function_pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\):\s*(?:"""|\'\'\')(.*?)(?:"""|\'\'\')(?:.*?)(?=def|\Z)'
                matches = re.findall(function_pattern, content, re.DOTALL)

                for match in matches:
                    function_name = match[0]
                    function_params = match[1]
                    docstring = match[2].strip()

                    pairs.append({
                        'function_name': function_name,
                        'params': function_params,
                        'docstring': docstring,
                        'file_path': file_path
                    })
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    return pairs #a list of dictionaries of function name, params, docstring and file path


def prepare_dataset(function_docstring_pairs):
    """Convert extracted pairs to a format suitable for training."""
    data = []

    for pair in function_docstring_pairs:
        # Construct input and target strings
        input_text = f"Generate documentation: def {pair['function_name']}({pair['params']}):" ###OUR PROMPT!!!CHANGE THIS###
        target_text = pair['docstring']

        data.append({
            'input': input_text,
            'target': target_text
        })

    import pandas as pd
    return pd.DataFrame(data) #a dataframe with input and target columns