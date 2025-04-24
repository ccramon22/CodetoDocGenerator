import re
import ast



class FunctionVisitor(ast.NodeVisitor):
    def __init__(self, file_path):
        self.functions = []
        self.file_path = file_path
        self.imports = []
        self.classes = []
        self.current_class = None

    def visit_Import(self, node):
        # Collect import statements
        for alias in node.names:
            self.imports.append(f"import {alias.name}")

    def visit_ImportFrom(self, node):
        # Collect from-import statements
        module = node.module
        for alias in node.names:
            self.imports.append(f"from {module} import {alias.name}")

    def visit_ClassDef(self, node):
        # Store class information
        class_info = {
            'name': node.name,
            'docstring': ast.get_docstring(node) or "",
            'methods': []
        }
        self.current_class = class_info
        self.classes.append(class_info)
        self.generic_visit(node)
        self.current_class = None

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

        # Get context information
        context = {
            'imports': self.imports.copy(),
            'class_context': self.current_class['name'] if self.current_class else None,
            'file_path': self.file_path
        }

        function_info = {
            'function_name': function_name,
            'params': params_str,
            'docstring': docstring,
            'function_body': function_body,
            'context': context
        }

        if self.current_class:
            self.current_class['methods'].append(function_info)
        else:
            self.functions.append(function_info)

        self.generic_visit(node)


def extract_function_docstring_pairs_ast(python_files):
    pairs = []

    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

                # Parse the file
                tree = ast.parse(content)

                # Visit all nodes
                visitor = FunctionVisitor(file_path)
                visitor.visit(tree)

                # Add functions with their context
                for func in visitor.functions:
                    pairs.append(func)

                # Add class methods with their context
                for class_info in visitor.classes:
                    for method in class_info['methods']:
                        method['context']['class_docstring'] = class_info['docstring']
                        pairs.append(method)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    return pairs


def prepare_dataset(pairs):
    """Convert pairs to DataFrame and save to JSON."""
    import pandas as pd
    import json
    import os
    
    # Convert pairs to DataFrame
    df = pd.DataFrame(pairs)
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save to JSON
    output_path = 'data/processed_dataset.json'
    df.to_json(output_path, orient='records', indent=2)
    print(f"Saved processed dataset to {output_path}")
    
    return df