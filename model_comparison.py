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
            "codet5-large": {
                "model_id": "Salesforce/codet5-large",
                "model_type": "t5",
                "max_length": 512
            },
            "llama3-8b": {
                "model_id": "meta-llama/Llama-3.1-8B",
                "model_type": "causal",
                "max_length": 2048
            },
            "mistral-7b-v0.3": {
                "model_id": "mistralai/Mistral-7B-v0.3",
                "model_type": "causal",
                "max_length": 2048
            },
            "starcoder2-7b": {
                "model_id": "bigcode/starcoder2-7b",
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
        print(f"\nLoading {model_name}...")

        try:
            if config["model_type"] == "t5":
                tokenizer = RobertaTokenizer.from_pretrained(config["model_id"], token=self.hf_token)
                model = T5ForConditionalGeneration.from_pretrained(config["model_id"], token=self.hf_token)
                model = model.to(self.device)
            else:
                tokenizer = AutoTokenizer.from_pretrained(config["model_id"], token=self.hf_token)
                # For large models, let accelerate handle device placement
                model = AutoModelForCausalLM.from_pretrained(
                    config["model_id"],
                    torch_dtype=torch.float16,
                    device_map="auto",
                    token=self.hf_token
                )
                # Don't move the model manually when using device_map

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

            # Free memory
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()

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

        # Generate visualizations
        self.visualize_results()

    def visualize_results(self):
        """Generate visualizations for the model comparison results"""
        summary_results_path = os.path.join(self.output_dir, "summary_results.json")
        if not os.path.exists(summary_results_path):
            print("Summary results not found. Please run the evaluation first.")
            return

        with open(summary_results_path, "r") as f:
            results = json.load(f)

        models = list(results.keys())

        # 1. Response Time Chart
        self._plot_response_time_chart(models, results)

        # 2. Model Size Chart
        self._plot_model_size_chart(models, results)

        # 3. Quality Metrics Radar Chart
        self._plot_quality_radar_chart(models, results)

        # 4. Efficiency Analysis
        self._plot_efficiency_chart(models, results)

        # 5. Metric Bar Charts
        metrics = ["avg_rouge1", "avg_rouge2", "avg_rougeL", "avg_bleu"]
        self._plot_metric_bar_charts(models, metrics, results)

        # 6. Trade-off Analysis
        self._plot_tradeoff_chart(models, results)

        print(f"Visualizations saved to {self.output_dir}")

    def _plot_response_time_chart(self, models, results):
        """Plot response time comparison chart"""
        plt.figure(figsize=(10, 6))
        response_times = [results[model]["avg_generation_time"] for model in models]

        plt.bar(models, response_times, color='skyblue')
        plt.title('Model Response Time Comparison')
        plt.xlabel('Models')
        plt.ylabel('Average Generation Time (seconds)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "response_time_chart.png"))
        plt.close()

    def _plot_model_size_chart(self, models, results):
        """Plot model size comparison chart"""
        plt.figure(figsize=(10, 6))
        sizes = [results[model]["model_size_mb"] for model in models]

        plt.bar(models, sizes, color='lightgreen')
        plt.title('Model Size Comparison')
        plt.xlabel('Models')
        plt.ylabel('Size (MB)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "model_size_chart.png"))
        plt.close()

    def _plot_quality_radar_chart(self, models, results):
        """Plot quality metrics radar chart"""
        metrics = ["avg_rouge1", "avg_rouge2", "avg_rougeL", "avg_bleu"]

        # Convert to matplotlib format for radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        for model in models:
            values = [results[model][metric] for metric in metrics]
            values += values[:1]  # Close the loop

            ax.plot(angles, values, linewidth=2, label=model)
            ax.fill(angles, values, alpha=0.1)

        ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
        ax.set_ylim(0, 1)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title('Model Quality Metrics Comparison', y=1.1)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "quality_radar_chart.png"))
        plt.close()

    def _plot_efficiency_chart(self, models, results):
        """Plot efficiency analysis chart"""
        plt.figure(figsize=(12, 8))

        # Calculate quality score (average of metrics)
        quality_scores = []
        size_efficiency = []
        time_efficiency = []

        for model in models:
            quality = (results[model]["avg_rouge1"] +
                       results[model]["avg_rouge2"] +
                       results[model]["avg_rougeL"] +
                       results[model]["avg_bleu"]) / 4

            quality_scores.append(quality)
            # Quality per MB (scaled)
            size_efficiency.append((quality / results[model]["model_size_mb"]) * 10000)
            # Quality per second (scaled)
            time_efficiency.append((quality / results[model]["avg_generation_time"]) * 100)

        # Plot as subplots
        plt.subplot(2, 1, 1)
        plt.bar(models, size_efficiency, color='coral')
        plt.title('Size Efficiency (Quality per MB)')
        plt.xticks(rotation=45)

        plt.subplot(2, 1, 2)
        plt.bar(models, time_efficiency, color='purple')
        plt.title('Time Efficiency (Quality per Second)')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "efficiency_chart.png"))
        plt.close()

    def _plot_metric_bar_charts(self, models, metrics, results):
        """Plot individual bar charts for each metric"""
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            values = [results[model][metric] for model in models]

            plt.bar(models, values, alpha=0.7)
            plt.title(f'{metric.replace("avg_", "").upper()} Score Comparison')
            plt.xlabel('Models')
            plt.ylabel('Score')
            plt.xticks(rotation=45)
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"{metric}_chart.png"))
            plt.close()

    def _plot_tradeoff_chart(self, models, results):
        """Plot trade-off analysis chart"""
        plt.figure(figsize=(12, 8))

        # Extract data
        sizes = [results[model]["model_size_mb"] for model in models]
        times = [results[model]["avg_generation_time"] for model in models]

        # Calculate quality scores
        quality_scores = []
        for model in models:
            quality = (results[model]["avg_rouge1"] +
                       results[model]["avg_rouge2"] +
                       results[model]["avg_rougeL"] +
                       results[model]["avg_bleu"]) / 4
            quality_scores.append(quality * 5000)  # Scale for visibility

        # Create scatter plot with custom sizes
        plt.scatter(times, sizes, s=quality_scores, alpha=0.6)

        # Add labels for each point
        for i, model in enumerate(models):
            plt.annotate(model, (times[i], sizes[i]),
                         textcoords="offset points",
                         xytext=(0, 10),
                         ha='center')

        plt.title('Model Trade-offs: Size vs Speed vs Quality')
        plt.xlabel('Response Time (seconds)')
        plt.ylabel('Model Size (MB)')
        plt.grid(True, linestyle='--', alpha=0.7)

        # Add a legend/explanation
        plt.figtext(0.5, 0.01,
                    "Bubble size represents quality score (larger is better)",
                    ha="center", fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "tradeoff_chart.png"))
        plt.close()


def main():
    comparer = ModelComparer()
    comparer.prepare_test_cases(num_examples=5)
    comparer.evaluate_models()
    # Visualization is now automatically called within evaluate_models


if __name__ == "__main__":
    main()