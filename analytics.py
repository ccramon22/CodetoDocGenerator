import time
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from transformers import TrainerCallback
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from datetime import datetime

class TrainingAnalytics:
    def __init__(self):
        self.start_time = time.time()
        self.metrics = {
            'loss': [],
            'learning_rate': [],
            'epoch': [],
            'step': [],
            'eval_loss': [],
            'eval_accuracy': [],
            'eval_precision': [],
            'eval_recall': [],
            'eval_f1': [],
            'timestamps': []
        }
        self.current_epoch = 0
        self.best_metrics = {
            'loss': float('inf'),
            'eval_loss': float('inf'),
            'eval_accuracy': 0,
            'eval_f1': 0
        }
        self.training_history = []
        os.makedirs('analytics', exist_ok=True)

    def log_metrics(self, metrics_dict):
        """Log metrics for each training step."""
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        for key, value in metrics_dict.items():
            if key in self.metrics:
                self.metrics[key].append(value)
        
        self.metrics['timestamps'].append(elapsed_time)
        self._update_best_metrics(metrics_dict)
        self._save_metrics()

    def _update_best_metrics(self, metrics_dict):
        """Update best metrics if current metrics are better."""
        for metric in ['loss', 'eval_loss', 'eval_accuracy', 'eval_f1']:
            if metric in metrics_dict:
                if 'loss' in metric:
                    if metrics_dict[metric] < self.best_metrics[metric]:
                        self.best_metrics[metric] = metrics_dict[metric]
                else:
                    if metrics_dict[metric] > self.best_metrics[metric]:
                        self.best_metrics[metric] = metrics_dict[metric]

    def _save_metrics(self):
        """Save metrics to JSON file."""
        metrics_path = 'analytics/training_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def end_training(self):
        """Finalize training analytics and generate visualizations."""
        self._generate_visualizations()
        self._save_training_summary()

    def _generate_visualizations(self):
        """Generate various training visualizations."""
        plt.style.use('seaborn')
        
        # Loss over time
        plt.figure(figsize=(12, 6))
        plt.plot(self.metrics['timestamps'], self.metrics['loss'], label='Training Loss')
        if self.metrics['eval_loss']:
            plt.plot(self.metrics['timestamps'], self.metrics['eval_loss'], label='Validation Loss')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss over Time')
        plt.legend()
        plt.savefig('analytics/loss_over_time.png')
        plt.close()

        # Learning rate schedule
        plt.figure(figsize=(12, 6))
        plt.plot(self.metrics['timestamps'], self.metrics['learning_rate'])
        plt.xlabel('Time (seconds)')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.savefig('analytics/learning_rate.png')
        plt.close()

        # Evaluation metrics
        if self.metrics['eval_accuracy']:
            plt.figure(figsize=(12, 6))
            metrics = ['eval_accuracy', 'eval_precision', 'eval_recall', 'eval_f1']
            for metric in metrics:
                if self.metrics[metric]:
                    plt.plot(self.metrics['timestamps'], self.metrics[metric], label=metric)
            plt.xlabel('Time (seconds)')
            plt.ylabel('Score')
            plt.title('Evaluation Metrics over Time')
            plt.legend()
            plt.savefig('analytics/evaluation_metrics.png')
            plt.close()

    def _save_training_summary(self):
        """Save a summary of the training process."""
        summary = {
            'total_training_time': time.time() - self.start_time,
            'best_metrics': self.best_metrics,
            'final_metrics': {k: v[-1] if v else None for k, v in self.metrics.items()},
            'total_steps': len(self.metrics['loss']),
            'timestamp': datetime.now().isoformat()
        }
        
        with open('analytics/training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

    def get_training_time(self):
        """Return formatted training time."""
        total_time = time.time() - self.start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        return f"{hours}h {minutes}m {seconds}s"

    def get_average_seconds_per_step(self):
        """Calculate average time per training step."""
        if not self.metrics['timestamps']:
            return 0
        return np.mean(np.diff(self.metrics['timestamps']))

class AnalyticsCallback(TrainerCallback):
    def __init__(self, analytics):
        self.analytics = analytics

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            self.analytics.log_metrics(logs)

    def on_epoch_end(self, args, state, control, **kwargs):
        self.analytics.current_epoch += 1