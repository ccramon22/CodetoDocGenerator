import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import os
import json
from transformers import TrainerCallback


class TrainingAnalytics:
    def __init__(self):
        self.start_time = time.time()
        self.end_time = None
        self.training_loss = []
        self.eval_loss = []
        self.learning_rates = []
        self.epochs = []
        self.steps = []
        self.checkpoint_sizes = []  # To track model size at checkpoints
        self.analytics_dir = "./analytics"

        # Create analytics directory
        if not os.path.exists(self.analytics_dir):
            os.makedirs(self.analytics_dir)

    def record_training_step(self, step, epoch, loss, learning_rate):
        """Record metrics from a training step"""
        self.steps.append(step)
        self.epochs.append(epoch)
        self.training_loss.append(loss)
        self.learning_rates.append(learning_rate)

    def record_eval_step(self, eval_loss):
        """Record metrics from an evaluation step"""
        self.eval_loss.append(eval_loss)

    def record_checkpoint(self, checkpoint_path):
        """Record the size of model checkpoint"""
        size_bytes = 0
        for dirpath, dirnames, filenames in os.walk(checkpoint_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                size_bytes += os.path.getsize(fp)

        # Convert to MB for easier visualization
        size_mb = size_bytes / (1024 * 1024)
        self.checkpoint_sizes.append((len(self.steps), size_mb))

    def end_training(self):
        """Mark the end of training and generate visualizations"""
        self.end_time = time.time()
        self.generate_visualizations()
        self.save_metrics()

    def get_training_time(self):
        """Return the total training time as a formatted string"""
        if not self.end_time:
            self.end_time = time.time()

        total_seconds = self.end_time - self.start_time
        return str(timedelta(seconds=int(total_seconds)))

    def get_average_seconds_per_step(self):
        """Calculate average seconds per training step"""
        if not self.end_time:
            self.end_time = time.time()

        total_seconds = self.end_time - self.start_time
        if len(self.steps) > 0:
            return total_seconds / len(self.steps)
        return 0

    def generate_visualizations(self):
        """Generate and save visualizations of training progress"""
        # Make sure we have some data to visualize
        if not self.steps:
            print("No training data available for visualization")
            return

        # Set a consistent style for better visuals
        plt.style.use('ggplot')

        # Figure 1: Training and Evaluation Loss with shaded confidence area
        plt.figure(figsize=(14, 8))
        plt.plot(self.steps, self.training_loss, 'b-', linewidth=2, label='Training Loss')

        # Add a smoothed trendline for training loss
        if len(self.training_loss) > 10:
            window_size = min(10, len(self.training_loss) // 5)
            smoothed_loss = np.convolve(self.training_loss, np.ones(window_size) / window_size, mode='valid')
            smoothed_steps = self.steps[window_size - 1:]
            plt.plot(smoothed_steps, smoothed_loss, 'r--', linewidth=2, label='Smoothed Trend')

        # Plot evaluation loss at corresponding steps with markers
        if self.eval_loss:
            # We need to map evaluation points to steps
            eval_indices = np.linspace(0, len(self.steps) - 1, len(self.eval_loss), dtype=int)
            eval_steps = [self.steps[i] for i in eval_indices]
            plt.plot(eval_steps, self.eval_loss, 'ro-', markersize=8, label='Evaluation Loss')

        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training and Evaluation Loss Over Time', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Add shaded area to highlight convergence region
        if len(self.training_loss) > 10:
            last_quarter = len(self.training_loss) // 4
            min_loss = min(self.training_loss[-last_quarter:])
            max_loss = max(self.training_loss[-last_quarter:])
            plt.axhspan(min_loss - 0.01, max_loss + 0.01, alpha=0.2, color='green', label='Convergence Region')

        plt.savefig(os.path.join(self.analytics_dir, 'loss_curve_enhanced.png'), dpi=300)

        # Figure 2: Learning Rate Decay with highlighted phases
        plt.figure(figsize=(14, 7))
        plt.semilogy(self.steps, self.learning_rates, 'g-', linewidth=2)

        # Highlight different phases of learning rate
        if len(self.learning_rates) > 10:
            # Find significant drops in learning rate
            lr_changes = []
            threshold = 0.1  # 10% change threshold

            for i in range(1, len(self.learning_rates)):
                if abs(self.learning_rates[i] - self.learning_rates[i - 1]) / self.learning_rates[i - 1] > threshold:
                    lr_changes.append(i)

            # Highlight regions between changes
            colors = ['#ffcccc', '#ccffcc', '#ccccff', '#ffffcc']
            for i in range(len(lr_changes) + 1):
                start = 0 if i == 0 else lr_changes[i - 1]
                end = len(self.steps) if i == len(lr_changes) else lr_changes[i]

                if end > start:
                    plt.axvspan(self.steps[start], self.steps[end - 1],
                                alpha=0.3, color=colors[i % len(colors)])

        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('Learning Rate (log scale)', fontsize=12)
        plt.title('Learning Rate Schedule with Training Phases', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.analytics_dir, 'learning_rate_enhanced.png'), dpi=300)

        # Figure 3: 3D visualization of loss landscape
        if len(self.steps) > 10 and len(self.eval_loss) > 2:
            try:
                from mpl_toolkits.mplot3d import Axes3D
                from scipy.interpolate import griddata

                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(111, projection='3d')

                # Create a meshgrid for the surface
                eval_indices = np.linspace(0, len(self.steps) - 1, len(self.eval_loss), dtype=int)
                eval_steps = [self.steps[i] for i in eval_indices]

                # Use epochs and steps to create a 2D grid
                X = np.array(eval_steps)
                Y = np.array(self.epochs)[eval_indices]
                Z = np.array(self.eval_loss)

                # Plot the actual evaluation points
                ax.scatter(X, Y, Z, c='r', marker='o', s=100, label='Evaluation Points')

                # Create an interpolated surface visualization
                if len(X) > 3:  # Need at least 4 points for reasonable interpolation
                    # Create a finer mesh grid for interpolation
                    xi = np.linspace(min(X), max(X), 20)
                    yi = np.linspace(min(Y), max(Y), 20)
                    Xi, Yi = np.meshgrid(xi, yi)

                    # Interpolate Z values on the mesh grid
                    Zi = griddata((X, Y), Z, (Xi, Yi), method='cubic')

                    # Plot the surface
                    surf = ax.plot_surface(Xi, Yi, Zi, cmap='viridis', alpha=0.7,
                                           linewidth=0, antialiased=True)

                    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

                ax.set_xlabel('Training Steps', fontsize=12)
                ax.set_ylabel('Epochs', fontsize=12)
                ax.set_zlabel('Loss', fontsize=12)
                ax.set_title('3D Loss Landscape', fontsize=16, fontweight='bold')
                plt.tight_layout()
                plt.savefig(os.path.join(self.analytics_dir, '3d_loss_landscape.png'), dpi=300)
            except ImportError:
                print("Could not create 3D visualization. Install scipy for this feature.")

        # Figure 4: Training progress radar chart
        if self.eval_loss:
            plt.figure(figsize=(10, 10))

            # Calculate metrics for radar chart
            n_metrics = 6

            # Calculate normalized values (0-1 scale)
            if len(self.training_loss) > 1:
                train_loss_improvement = max(0, 1 - (self.training_loss[-1] / self.training_loss[0]))
            else:
                train_loss_improvement = 0

            if len(self.eval_loss) > 1:
                eval_loss_improvement = max(0, 1 - (self.eval_loss[-1] / self.eval_loss[0]))
            else:
                eval_loss_improvement = 0

            # Learning rate adaptation (how much it decreased)
            if len(self.learning_rates) > 1:
                lr_adaptation = max(0, 1 - (self.learning_rates[-1] / self.learning_rates[0]))
            else:
                lr_adaptation = 0

            # Training stability (inverse of variance in last quarter)
            if len(self.training_loss) > 4:
                last_quarter = self.training_loss[-len(self.training_loss) // 4:]
                stability = 1 - min(1, np.std(last_quarter) / np.mean(last_quarter))
            else:
                stability = 0.5  # Default value

            # Training duration normalized to 0-1 (assuming longer is better, up to a point)
            duration = min(1.0, (self.end_time - self.start_time) / (60 * 60 * 24))  # Normalize to 0-1 over 24 hours

            # Convergence (how flat the last part of the curve is)
            if len(self.eval_loss) > 3:
                last_changes = np.abs(np.diff(self.eval_loss[-3:]))
                convergence = 1 - min(1, np.mean(last_changes) / np.mean(self.eval_loss[-3:]))
            else:
                convergence = 0.5  # Default value

            metrics = [
                train_loss_improvement,
                eval_loss_improvement,
                lr_adaptation,
                stability,
                duration,
                convergence
            ]

            labels = [
                'Training Loss\nImprovement',
                'Evaluation Loss\nImprovement',
                'Learning Rate\nAdaptation',
                'Training\nStability',
                'Training\nDuration',
                'Model\nConvergence'
            ]

            # Plot the radar chart
            angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
            metrics += metrics[:1]  # Close the loop
            angles += angles[:1]  # Close the loop

            ax = plt.subplot(111, polar=True)
            plt.xticks(angles[:-1], labels, fontsize=14)
            ax.set_rlabel_position(0)
            plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=12)
            plt.ylim(0, 1)

            # Plot metrics and fill area
            ax.plot(angles, metrics, 'o-', linewidth=2, color='#FF5722')
            ax.fill(angles, metrics, alpha=0.25, color='#FF5722')

            plt.title('Training Performance Metrics', size=16, y=1.1, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(self.analytics_dir, 'training_radar.png'), dpi=300)

        # Figure 5: Combined dashboard with all important metrics
        plt.figure(figsize=(20, 15))
        plt.suptitle('Model Training Dashboard', fontsize=20, fontweight='bold', y=0.98)

        # Training and eval loss
        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(self.steps, self.training_loss, 'b-', linewidth=2, label='Training Loss')

        if self.eval_loss:
            eval_indices = np.linspace(0, len(self.steps) - 1, len(self.eval_loss), dtype=int)
            eval_steps = [self.steps[i] for i in eval_indices]
            ax1.plot(eval_steps, self.eval_loss, 'ro-', markersize=6, label='Evaluation Loss')

        ax1.set_xlabel('Training Steps', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Loss Curves', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Learning rate
        ax2 = plt.subplot(2, 2, 2)
        ax2.semilogy(self.steps, self.learning_rates, 'g-', linewidth=2)
        ax2.set_xlabel('Training Steps', fontsize=12)
        ax2.set_ylabel('Learning Rate (log scale)', fontsize=12)
        ax2.set_title('Learning Rate Schedule', fontsize=14)
        ax2.grid(True, alpha=0.3)

        # Model size
        ax3 = plt.subplot(2, 2, 3)
        if self.checkpoint_sizes:
            steps, sizes = zip(*self.checkpoint_sizes)
            ax3.plot(steps, sizes, 'mo-', linewidth=2)
            ax3.set_xlabel('Training Steps', fontsize=12)
            ax3.set_ylabel('Model Size (MB)', fontsize=12)
            ax3.set_title('Model Size Evolution', fontsize=14)
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No checkpoint size data available',
                     horizontalalignment='center', verticalalignment='center',
                     transform=ax3.transAxes, fontsize=12)

        # Training statistics
        ax4 = plt.subplot(2, 2, 4)
        ax4.axis('off')

        if self.training_loss:
            # Calculate key statistics
            initial_loss = self.training_loss[0]
            final_loss = self.training_loss[-1]
            loss_reduction = ((initial_loss - final_loss) / initial_loss) * 100

            if self.eval_loss:
                eval_indices = np.linspace(0, len(self.steps) - 1, len(self.eval_loss), dtype=int)
                final_eval = self.eval_loss[-1]
                best_eval = min(self.eval_loss)
                best_epoch = self.epochs[int(eval_indices[np.argmin(self.eval_loss)])]
            else:
                final_eval = "N/A"
                best_eval = "N/A"
                best_epoch = "N/A"

            total_time = self.get_training_time()
            avg_time_per_step = f"{self.get_average_seconds_per_step():.2f}s"

            # Create a text summary
            stats_text = [
                f"Training Summary Statistics:",
                f"───────────────────────────",
                f"Initial Training Loss: {initial_loss:.4f}",
                f"Final Training Loss: {final_loss:.4f}",
                f"Loss Reduction: {loss_reduction:.2f}%",
                f"Best Evaluation Loss: {best_eval if isinstance(best_eval, str) else best_eval:.4f}",
                f"Best Epoch: {best_epoch}",
                f"Final Evaluation Loss: {final_eval if isinstance(final_eval, str) else final_eval:.4f}",
                f"Total Training Time: {total_time}",
                f"Avg. Time per Step: {avg_time_per_step}",
                f"Total Steps: {len(self.steps)}",
                f"───────────────────────────",
            ]

            y_pos = 0.95
            for line in stats_text:
                ax4.text(0.05, y_pos, line, fontsize=12, va='top', ha='left')
                y_pos -= 0.08

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(self.analytics_dir, 'training_dashboard.png'), dpi=300)

        # Original visualizations (kept for backward compatibility)
        # Figure 1: Training and Evaluation Loss
        plt.figure(figsize=(12, 8))
        plt.plot(self.steps, self.training_loss, label='Training Loss')

        # Plot evaluation loss at corresponding steps
        if self.eval_loss:
            # We need to map evaluation points to steps
            eval_steps = [self.steps[i] for i in
                          range(0, len(self.steps), len(self.steps) // max(1, len(self.eval_loss)))][
                         :len(self.eval_loss)]
            plt.plot(eval_steps, self.eval_loss, 'ro-', label='Evaluation Loss')

        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Training and Evaluation Loss Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.analytics_dir, 'loss_curve.png'))

        # Figure 2: Learning Rate Decay
        plt.figure(figsize=(12, 6))
        plt.plot(self.steps, self.learning_rates)
        plt.xlabel('Training Steps')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True)
        plt.savefig(os.path.join(self.analytics_dir, 'learning_rate.png'))

        # Figure 3: Model Size Growth (if checkpoints were recorded)
        if self.checkpoint_sizes:
            steps, sizes = zip(*self.checkpoint_sizes)
            plt.figure(figsize=(10, 6))
            plt.plot(steps, sizes, 'bo-')
            plt.xlabel('Training Steps')
            plt.ylabel('Model Size (MB)')
            plt.title('Model Checkpoint Size Over Training')
            plt.grid(True)
            plt.savefig(os.path.join(self.analytics_dir, 'model_size.png'))

        # Figure 4: Combined metrics
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 1, 1)
        plt.plot(self.steps, self.training_loss, 'b-', label='Training Loss')
        if self.eval_loss:
            eval_steps = [self.steps[i] for i in
                          range(0, len(self.steps), len(self.steps) // max(1, len(self.eval_loss)))][
                         :len(self.eval_loss)]
            plt.plot(eval_steps, self.eval_loss, 'r-', label='Evaluation Loss')
        plt.title('Training Progress')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(self.steps, self.learning_rates, 'g-')
        plt.xlabel('Training Steps')
        plt.ylabel('Learning Rate')
        plt.grid(True)

        plt.savefig(os.path.join(self.analytics_dir, 'training_summary.png'))

    def save_metrics(self):
        """Save all tracked metrics to a JSON file for later analysis"""
        metrics = {
            'training_time_seconds': self.end_time - self.start_time,
            'training_time_formatted': self.get_training_time(),
            'steps': self.steps,
            'epochs': self.epochs,
            'training_loss': self.training_loss,
            'eval_loss': self.eval_loss,
            'learning_rates': self.learning_rates,
            'checkpoint_sizes': self.checkpoint_sizes
        }

        # Convert numpy values to native Python types for JSON serialization
        for key, value in metrics.items():
            if isinstance(value, list) and value and isinstance(value[0], np.number):
                metrics[key] = [float(v) if isinstance(v, np.number) else v for v in value]

        with open(os.path.join(self.analytics_dir, 'training_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"Analytics dashboard created in {self.analytics_dir} directory")


class AnalyticsCallback(TrainerCallback):
    def __init__(self, analytics):
        self.analytics = analytics

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % 100 == 0:  # Record every 100 steps
            loss = state.log_history[-1].get('loss', None) if state.log_history else None
            lr = state.log_history[-1].get('learning_rate', None) if state.log_history else None

            if loss is not None and lr is not None:
                self.analytics.record_training_step(
                    step=state.global_step,
                    epoch=state.epoch,
                    loss=loss,
                    learning_rate=lr
                )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and 'eval_loss' in metrics:
            self.analytics.record_eval_step(metrics['eval_loss'])

    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.exists(checkpoint_path):
            self.analytics.record_checkpoint(checkpoint_path)