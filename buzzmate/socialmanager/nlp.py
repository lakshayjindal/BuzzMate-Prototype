from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model (this step can be done once to avoid reloading each time)
model_name = "facebook/llama-7b"  # Replace with the exact model you're using
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
token="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"

def generate_response(input_text):
    """
    Generates a response from the LLaMA model based on the input text.

    Args:
    - input_text (str): The input text for which a response is generated.

    Returns:
    - str: The generated response.
    """
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt")

    # Generate a response from the model
    outputs = model.generate(inputs["input_ids"], max_length=100)

    # Decode the generated response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response

# Example usage
if __name__ == "__main__":
    input_message = "What is artificial intelligence?"
    response = generate_response(input_message)
    print("Response from LLaMA:", response)
