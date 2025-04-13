import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import time
import os
import seaborn as sns
from transformers import TrainerCallback


class ModelAnalytics:
    """Class for tracking, analyzing, and visualizing model performance."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.training_history = {
            'loss': [],
            'eval_loss': [],
            'learning_rate': [],
            'epoch': [],
            'timestamp': [],
            'seconds_per_step': [],
            'steps': []
        }
        self.start_time = None
        self.end_time = None
        self.last_log_time = None
        self.last_step = 0
        self.prediction_examples = []
        self.samples_analyzed = 0

    def start_tracking(self):
        """Start timing the training process."""
        self.start_time = time.time()
        self.last_log_time = self.start_time

    def end_tracking(self):
        """End timing the training process."""
        self.end_time = time.time()

    def log_metrics(self, logs: Dict[str, float], epoch: float):
        """Log training metrics at each step."""
        current_time = time.time()

        # Get step info if available
        current_step = logs.get('step', self.last_step + 1)
        steps_taken = current_step - self.last_step

        # Log metrics
        self.training_history['loss'].append(logs.get('loss', 0))
        self.training_history['eval_loss'].append(logs.get('eval_loss', 0))
        self.training_history['learning_rate'].append(logs.get('learning_rate', 0))
        self.training_history['epoch'].append(epoch)
        self.training_history['timestamp'].append(current_time)
        self.training_history['steps'].append(current_step)

        # Calculate seconds per step
        if self.last_log_time and steps_taken > 0:
            time_diff = current_time - self.last_log_time
            seconds_per_step = time_diff / steps_taken
            self.training_history['seconds_per_step'].append(seconds_per_step)
        else:
            self.training_history['seconds_per_step'].append(0)

        self.last_log_time = current_time
        self.last_step = current_step

    def log_prediction(self, code_snippet: str, actual_docstring: str, predicted_docstring: str):
        """Log a prediction example for later analysis."""
        self.prediction_examples.append({
            'code': code_snippet,
            'actual': actual_docstring,
            'predicted': predicted_docstring
        })
        self.samples_analyzed += 1

    def get_training_time(self) -> str:
        """Get the total training time in a human-readable format."""
        if not self.start_time or not self.end_time:
            return "Training not completed"

        total_seconds = self.end_time - self.start_time
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)

        return f"{hours}h {minutes}m {seconds}s"

    def get_average_seconds_per_step(self) -> float:
        """Calculate the average seconds per iteration."""
        if not self.training_history['seconds_per_step']:
            return 0.0
        valid_times = [t for t in self.training_history['seconds_per_step'] if t > 0]
        if not valid_times:
            return 0.0
        return sum(valid_times) / len(valid_times)

    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot the training and evaluation loss over epochs."""
        plt.figure(figsize=(12, 6))

        # Create a DataFrame for easier plotting
        history_df = pd.DataFrame({
            'epoch': self.training_history['epoch'],
            'loss': self.training_history['loss'],
            'eval_loss': self.training_history['eval_loss']
        })

        # Plot training loss
        plt.subplot(1, 2, 1)
        plt.plot(history_df['epoch'], history_df['loss'], label='Training Loss')
        if any(history_df['eval_loss']):
            plt.plot(history_df['epoch'], history_df['eval_loss'], label='Validation Loss')
        plt.title('Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        # Plot learning rate
        plt.subplot(1, 2, 2)
        plt.plot(self.training_history['epoch'], self.training_history['learning_rate'])
        plt.title('Learning Rate Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)

        plt.show()

    def plot_training_speed(self, save_path: Optional[str] = None):
        """Plot the training speed (seconds per iteration) over epochs."""
        if not self.training_history['seconds_per_step']:
            print("No training speed data available")
            return

        valid_speeds = [(epoch, speed) for epoch, speed in
                        zip(self.training_history['epoch'], self.training_history['seconds_per_step'])
                        if speed > 0]

        if not valid_speeds:
            print("No valid training speed data available")
            return

        epochs, speeds = zip(*valid_speeds)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, speeds)
        plt.title('Training Speed Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Seconds per Iteration')
        plt.grid(True, linestyle='--', alpha=0.7)

        # Add average line
        avg_speed = self.get_average_seconds_per_step()
        plt.axhline(avg_speed, color='r', linestyle='--',
                    label=f'Average: {avg_speed:.2f} s/it')
        plt.legend()

        if save_path:
            plt.savefig(save_path)

        plt.show()

    def generate_summary_stats(self) -> Dict[str, Any]:
        """Generate summary statistics for the model."""
        if not self.training_history['loss']:
            return {"error": "No training data available"}

        stats = {
            "model_name": self.model_name,
            "training_time": self.get_training_time(),
            "epochs": max(self.training_history['epoch']) if self.training_history['epoch'] else 0,
            "final_training_loss": self.training_history['loss'][-1] if self.training_history['loss'] else None,
            "final_eval_loss": self.training_history['eval_loss'][-1] if self.training_history['eval_loss'] else None,
            "average_seconds_per_iteration": self.get_average_seconds_per_step(),
            "samples_analyzed": self.samples_analyzed
        }

        return stats

    def plot_prediction_quality(self, save_path: Optional[str] = None):
        """Plot analysis of prediction quality."""
        if not self.prediction_examples:
            print("No prediction examples to analyze")
            return

        # Create a simple scoring system for examples (placeholder - you might want to use BLEU, ROUGE, etc.)
        similarity_scores = []
        for example in self.prediction_examples:
            # Very simple similarity calculation (for demo purposes)
            # In real life, you'd use a proper NLP metric
            actual_words = set(example['actual'].lower().split())
            predicted_words = set(example['predicted'].lower().split())
            if not actual_words:
                similarity = 0.0
            else:
                similarity = len(actual_words.intersection(predicted_words)) / len(actual_words)
            similarity_scores.append(similarity * 100)  # Convert to percentage

        plt.figure(figsize=(10, 6))
        plt.hist(similarity_scores, bins=10, alpha=0.7)
        plt.title('Distribution of Prediction Quality Scores')
        plt.xlabel('Similarity Score (%)')
        plt.ylabel('Number of Examples')
        plt.grid(True, linestyle='--', alpha=0.5)

        # Add mean score line
        mean_score = np.mean(similarity_scores)
        plt.axvline(mean_score, color='r', linestyle='--', label=f'Mean Score: {mean_score:.2f}%')
        plt.legend()

        if save_path:
            plt.savefig(save_path)

        plt.show()

    def create_presentation_dashboard(self, output_dir: str = "./analytics"):
        """Create a comprehensive dashboard for presentation."""
        os.makedirs(output_dir, exist_ok=True)

        # Generate all analytics
        stats = self.generate_summary_stats()

        # Save plots
        self.plot_training_history(os.path.join(output_dir, "training_history.png"))
        self.plot_training_speed(os.path.join(output_dir, "training_speed.png"))
        self.plot_prediction_quality(os.path.join(output_dir, "prediction_quality.png"))

        # Create summary table
        stats_df = pd.DataFrame([stats])
        stats_df.to_csv(os.path.join(output_dir, "model_summary.csv"), index=False)

        # Create example predictions table
        if self.prediction_examples:
            examples_df = pd.DataFrame(self.prediction_examples[:10])  # First 10 examples
            examples_df.to_csv(os.path.join(output_dir, "prediction_examples.csv"), index=False)

        print(f"Analytics dashboard created in {output_dir}")

        return stats


class LoggingCallback(TrainerCallback):
    """Callback to log metrics during training."""

    def __init__(self, analytics):
        self.analytics = analytics

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when the trainer logs metrics."""
        if logs:
            self.analytics.log_metrics(logs, state.epoch)