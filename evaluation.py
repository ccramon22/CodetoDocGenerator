def generate_documentation(code_snippet, model, tokenizer, max_length=150):
    """
    Generate documentation for a given code snippet.
    The code_snippet should include both the function signature and body.
    """
    input_text = f"Generate documentation: {code_snippet}"

    # Tokenize the input
    input_ids = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).input_ids

    # Generate output
    outputs = model.generate(
        input_ids,
        max_length=max_length,
        num_beams=4,
        early_stopping=True
    )

    # Decode and return
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


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