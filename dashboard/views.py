from django.shortcuts import render
from django.http import JsonResponse
from main import main as run_training
from model import load_model_and_tokenizer
from evaluation import generate_documentation

def home(request):
    return render(request, 'home.html')

def train_model(request):
    if request.method == "POST":
        run_training()
        return render(request, 'train.html', {"status": "Model trained!"})
    return render(request, 'train.html')

def generate_doc(request):
    if request.method == "POST":
        code = request.POST.get("code")
        if not code:
            return JsonResponse({"doc": "No code provided."})
        model, tokenizer = load_model_and_tokenizer()
        doc = generate_documentation(code, model, tokenizer)
        return JsonResponse({"doc": doc})
    return render(request, 'generate.html')
