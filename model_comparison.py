import os
import time
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    T5ForConditionalGeneration, RobertaTokenizer
)
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
import pandas as pd
from tqdm import tqdm
import gc


class ModelComparer:
    def __init__(self, output_dir="./model_comparison_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Check for GPU availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Define models to compare
        self.models_config = {
            "codet5-base": {
                "model_id": "Salesforce/codet5-base",
                "model_type": "t5",
                "max_length": 512
            },
            "llama3-8b": {
                "model_id": "meta-llama/Meta-Llama-3-8B",
                "model_type": "causal",
                "max_length": 2048
            },
            "mistral-7b-v0.3": {
                "model_id": "mistralai/Mistral-7B-v0.3",
                "model_type": "causal",
                "max_length": 2048
            }
        }

        # Evaluation results
        self.results = {model: {} for model in self.models_config.keys()}
        self.test_cases = []

        # Evaluation metrics
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.bleu = BLEU()

        # Hugging Face token
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if not self.hf_token:
            raise ValueError("Hugging Face token not found. Please set the HUGGINGFACE_TOKEN environment variable.")

    def load_model(self, model_name):
        """Load a specific model and tokenizer"""
        config = self.models_config[model_name]
        print(f"Loading {model_name}...")

        try:
            if config["model_type"] == "t5":
                tokenizer = RobertaTokenizer.from_pretrained(config["model_id"], use_auth_token=self.hf_token)
                model = T5ForConditionalGeneration.from_pretrained(config["model_id"], use_auth_token=self.hf_token)
            else:
                tokenizer = AutoTokenizer.from_pretrained(config["model_id"], use_auth_token=self.hf_token)
                model = AutoModelForCausalLM.from_pretrained(
                    config["model_id"],
                    torch_dtype=torch.float16,
                    device_map="auto",
                    use_auth_token=self.hf_token
                )

            model = model.to(self.device)
            model_size_mb = self._get_model_size(model)
            self.results[model_name]["model_size_mb"] = model_size_mb
            print(f"Model size: {model_size_mb:.2f} MB")

            return model, tokenizer
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            return None, None

    def _get_model_size(self, model):
        """Calculate model size in MB"""
        param_size = sum(param.nelement() * param.element_size() for param in model.parameters())
        buffer_size = sum(buffer.nelement() * buffer.element_size() for buffer in model.buffers())
        return (param_size + buffer_size) / 1024 ** 2

    def prepare_test_cases(self, num_examples=10):
        """Prepare test cases with non-descriptive function names"""
        self.test_cases = [
            {
                "code": "def func1(numbers):\n    total = sum(numbers)\n    return total / len(numbers)",
                "reference_doc": "Calculate the average of a list of numbers.\n\nParameters:\n    numbers (list): A list of numbers\n\nReturns:\n    float: The average of the numbers"
            },
            {
                "code": "def func2(dict1, dict2):\n    result = dict1.copy()\n    result.update(dict2)\n    return result",
                "reference_doc": "Merge two dictionaries into a new dictionary.\n\nParameters:\n    dict1 (dict): The first dictionary\n    dict2 (dict): The second dictionary\n\nReturns:\n    dict: A new dictionary containing all key-value pairs from both input dictionaries"
            },
        ][:num_examples]

    def generate_documentation(self, model_name, code):
        """Generate documentation for a code snippet using the specified model"""
        config = self.models_config[model_name]
        model, tokenizer = self.load_model(model_name)

        if model is None:
            return "Error: Model could not be loaded", 0

        try:
            start_time = time.time()

            if config["model_type"] == "t5":
                input_text = f"Generate documentation for this code: {code}"
                input_ids = tokenizer(input_text, return_tensors="pt", max_length=config["max_length"],
                                      truncation=True).input_ids.to(self.device)

                with torch.no_grad():
                    outputs = model.generate(input_ids, max_length=150, num_beams=4, early_stopping=True)
                generated_doc = tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                prompt = f"""Write a complete Python documentation string (docstring) based only on analyzing the following code:

```python
{code}
Documentation:
"""
                inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7, do_sample=True, top_p=0.95)
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated_doc = generated_text.split("Documentation:")[1].strip()

            generation_time = time.time() - start_time
            return generated_doc, generation_time
        except Exception as e:
            print(f"Error generating documentation with {model_name}: {e}")
            return f"Error: {str(e)}", 0

    def evaluate_models(self, models_to_evaluate=None):
        """Evaluate all models on the test cases"""
        if models_to_evaluate is None:
            models_to_evaluate = list(self.models_config.keys())

        if not self.test_cases:
            self.prepare_test_cases()

        results_df = []

        for model_name in models_to_evaluate:
            print(f"\nEvaluating {model_name}...")
            model_results = {
                "model_name": model_name,
                "rouge1_scores": [],
                "rouge2_scores": [],
                "rougeL_scores": [],
                "bleu_scores": [],
                "generation_times": []
            }

            for i, test_case in enumerate(tqdm(self.test_cases)):
                code = test_case["code"]
                reference_doc = test_case["reference_doc"]

                generated_doc, generation_time = self.generate_documentation(model_name, code)

                rouge_scores = self.rouge_scorer.score(reference_doc, generated_doc)
                model_results["rouge1_scores"].append(rouge_scores["rouge1"].fmeasure)
                model_results["rouge2_scores"].append(rouge_scores["rouge2"].fmeasure)
                model_results["rougeL_scores"].append(rouge_scores["rougeL"].fmeasure)

                bleu_score = self.bleu.sentence_score(generated_doc, [reference_doc]).score / 100.0
                model_results["bleu_scores"].append(bleu_score)
                model_results["generation_times"].append(generation_time)

                results_df.append({
                    "model": model_name,
                    "example_id": i,
                    "code": code,
                    "reference_doc": reference_doc,
                    "generated_doc": generated_doc,
                    "generation_time": generation_time,
                    "rouge1": rouge_scores["rouge1"].fmeasure,
                    "rouge2": rouge_scores["rouge2"].fmeasure,
                    "rougeL": rouge_scores["rougeL"].fmeasure,
                    "bleu": bleu_score
                })

            self.results[model_name].update({
                "avg_rouge1": np.mean(model_results["rouge1_scores"]),
                "avg_rouge2": np.mean(model_results["rouge2_scores"]),
                "avg_rougeL": np.mean(model_results["rougeL_scores"]),
                "avg_bleu": np.mean(model_results["bleu_scores"]),
                "avg_generation_time": np.mean(model_results["generation_times"]),
            })

        results_df = pd.DataFrame(results_df)
        results_df.to_csv(os.path.join(self.output_dir, "detailed_results.csv"), index=False)

        with open(os.path.join(self.output_dir, "summary_results.json"), "w") as f:
            json.dump(self.results, f, indent=2)

    def visualize_results(self):
        """Generate visualizations for the model comparison results"""
        summary_results_path = os.path.join(self.output_dir, "summary_results.json")
        if not os.path.exists(summary_results_path):
            print("Summary results not found. Please run the evaluation first.")
            return

        with open(summary_results_path, "r") as f:
            results = json.load(f)

        models = list(results.keys())
        metrics = ["avg_rouge1", "avg_rouge2", "avg_rougeL", "avg_bleu", "avg_generation_time"]

        data = {metric: [results[model][metric] for model in models] for metric in metrics}

        self._plot_bar_chart(models, metrics, data)

    def _plot_bar_chart(self, models, metrics, data):
        """Plot a bar chart with metrics"""
        for metric in metrics:
            values = data[metric]
            plt.bar(models, values, alpha=0.7, label=metric)

        plt.title("Model Comparison Bar Chart")
        plt.xlabel("Models")
        plt.ylabel("Scores")
        plt.legend()
        plt.show()


def main():
    comparer = ModelComparer()
    comparer.prepare_test_cases(num_examples=5)
    comparer.evaluate_models()
    comparer.visualize_results()


if __name__ == "__main__":
    main()