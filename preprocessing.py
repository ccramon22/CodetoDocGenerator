import re
import ast
import astpretty



class FunctionVisitor(ast.NodeVisitor):
    def __init__(self):
        self.functions = []

    def visit_FunctionDef(self, node):
        # Extract function name
        function_name = node.name

        # Extract parameters
        params = []
        for arg in node.args.args:
            params.append(arg.arg)
        params_str = ', '.join(params)

        # Extract docstring
        docstring = ast.get_docstring(node) or ""

        # Extract function body
        body_lines = []
        for body_node in node.body:
            if not isinstance(body_node, ast.Expr) or not isinstance(body_node.value, ast.Str):
                # Skip the docstring node
                body_lines.append(ast.unparse(body_node))

        function_body = '\n'.join(body_lines)

        self.functions.append({
            'function_name': function_name,
            'params': params_str,
            'docstring': docstring,
            'function_body': function_body
        })

        self.generic_visit(node)


def extract_function_docstring_pairs_ast(python_files):
    pairs = []

    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

                # Parse the file
                tree = ast.parse(content)
                astpretty.pprint(tree)

                # Visit all function nodes
                visitor = FunctionVisitor()
                visitor.visit(tree)

                for func in visitor.functions:
                    func['file_path'] = file_path
                    pairs.append(func)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    return pairs


def prepare_dataset(pairs):
    """Convert function-docstring pairs to a DataFrame."""
    import pandas as pd

    # Convert to DataFrame
    df = pd.DataFrame(pairs)

    # Create input-output pairs for the model
    df['input'] = df.apply(
        lambda row: f"Code: def {row['function_name']}({row['params']}):\n{row['function_body']}\nDocumentation:",
        axis=1
    )

    df['output'] = df['docstring']

    # Filter out entries with empty docstrings
    df = df[df['docstring'].str.strip() != '']

    # Sample a manageable subset (important to reduce memory usage)
    if len(df) > 10000:
        df = df.sample(10000, random_state=42)

    return df