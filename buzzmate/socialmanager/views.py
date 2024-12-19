from django.shortcuts import render
from django.http import JsonResponse
from transformers import AutoTokenizer, AutoModelForCausalLM
# from .nlp import generate_response  # Assuming an NLP utility module

HUGGINGFACE_API_TOKEN = "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"
model_name = "meta-llama/Llama-3.3-70B-Instruct"  # Use the exact model you want
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HUGGINGFACE_API_TOKEN)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=HUGGINGFACE_API_TOKEN)

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

def generate_view(request):
    """
    A view to receive an input message and generate a response from the LLaMA model.

    Args:
    - request (HttpRequest): The incoming request.

    Returns:
    - JsonResponse: The response from the LLaMA model.
    """
    # Get the input message from the GET or POST request
    user_message = request.GET.get('message') or request.POST.get('message')

    if user_message:
        # Generate a response using the generate_response function
        response = generate_response(user_message)
        return JsonResponse({"response": response})
    else:
        return JsonResponse({"error": "No message provided"}, status=400)
