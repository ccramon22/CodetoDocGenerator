def generate_documentation(code_snippet, model, tokenizer, max_length=150):
    """
    Generate documentation for a given code snippet using a one-shot and chain-of-thought approach.
    The code_snippet should include both the function signature and body.
    """
    # Examples of good documentation for reference
    example1_code = """def calculate_average(numbers):
    total = sum(numbers)
    return total / len(numbers)"""
    
    example1_doc = """This function calculates the average of a list of numbers.

Parameters:
    numbers (list): A list of numeric values

Returns:
    float: The arithmetic mean of the numbers

Example:
    >>> calculate_average([1, 2, 3, 4, 5])
    3.0

The function works by:
1. Summing all numbers in the input list
2. Dividing the sum by the number of elements
3. Returning the result as a float"""

    example2_code = """def process_data(data, threshold=0.5):
    if not isinstance(data, list):
        raise TypeError("Input must be a list")
    filtered = [x for x in data if x > threshold]
    return sorted(filtered)"""
    
    example2_doc = """Process and filter a list of numerical data.

This function takes a list of numbers and returns a sorted list of values
that exceed the specified threshold. It includes input validation and
error handling.

Parameters:
    data (list): List of numerical values to process
    threshold (float, optional): Minimum value to include in results.
                                Defaults to 0.5.

Returns:
    list: Sorted list of values greater than the threshold

Raises:
    TypeError: If input data is not a list

Example:
    >>> process_data([0.1, 0.6, 0.3, 0.8, 0.4])
    [0.6, 0.8]"""

    prompt = f"""Let me show you how to write comprehensive Python documentation. Here are two examples:

Example 1 - Simple Function:
Code:
{example1_code}

Documentation:
{example1_doc}

Example 2 - Complex Function:
Code:
{example2_code}

Documentation:
{example2_doc}

Now, let's analyze how to write good documentation:

1. First, understand the function's purpose and behavior
   - What does the function do?
   - What problem does it solve?
   - Are there any special cases or edge conditions?

2. Document the interface
   - List all parameters with their types and descriptions
   - Document any optional parameters and their defaults
   - Describe the return value and its type
   - List any exceptions that might be raised

3. Add practical information
   - Include a clear example showing usage
   - Explain any important implementation details
   - Note any performance considerations
   - Mention any dependencies or requirements

4. Follow Python documentation standards
   - Use clear, concise language
   - Include type hints in the docstring
   - Format examples using doctest format
   - Use proper section headers

Now, let's document this new code:

Code:
{code_snippet}

Let's analyze this code step by step:
1. What is the function's main purpose and behavior?
2. What are the parameters and their types?
3. What does the function return?
4. Are there any special cases or edge conditions?
5. What's a good example of using this function?
6. How does the implementation work?

Documentation:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=300,  # Increased for more detailed documentation
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text.split("Documentation:")[1].strip() if "Documentation:" in generated_text else generated_text


def evaluate_model(test_snippets, model, tokenizer):
    """Evaluate model on test snippets."""
    results = []

    for snippet in test_snippets:
        generated_doc = generate_documentation(snippet, model, tokenizer)
        results.append({
            'code_snippet': snippet,
            'generated_documentation': generated_doc
        })

    return results