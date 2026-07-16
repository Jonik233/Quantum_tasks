import os
from config import label_names
import matplotlib.pyplot as plt


def compute_metrics(evaluator, epoch_predictions, epoch_labels):
    """
    Computes seqeval metrics for an entire epoch.
    epoch_predictions: List of numpy arrays of shape (batch_size, seq_len)
    epoch_labels: List of numpy arrays of shape (batch_size, seq_len)
    """
    mapped_predictions = []
    mapped_labels = []

    # Iterate through each batch in the epoch
    for batch_predictions, batch_labels in zip(epoch_predictions, epoch_labels):

        # Iterate through each sequence in the batch
        for seq_predictions, seq_labels in zip(batch_predictions, batch_labels):
            seq_pred_mapped = []
            seq_label_mapped = []

            # Iterate through each token in the sequence
            for pred, label in zip(seq_predictions, seq_labels):
                if label != -100:  # Ignore special tokens/padding
                    seq_pred_mapped.append(label_names[pred])
                    seq_label_mapped.append(label_names[label])

            mapped_predictions.append(seq_pred_mapped)
            mapped_labels.append(seq_label_mapped)

    # Epoch metrics
    metrics = evaluator.compute(predictions=mapped_predictions, references=mapped_labels)
    return metrics


def print_metrics(loss, metrics: dict, phase: str):
    """
    Prints the calculated metrics.
    """
    print(f"\n\n--- {phase} Metrics ---")
    print(f"Loss: {loss:.4f}")
    print(f"Overall Precision: {metrics.get('overall_precision', 0):.4f}")
    print(f"Overall Recall:    {metrics.get('overall_recall', 0):.4f}")
    print(f"Overall F1:        {metrics.get('overall_f1', 0):.4f}")
    print(f"Overall Accuracy:  {metrics.get('overall_accuracy', 0):.4f}")

    # Entity-level metrics
    if "MOUNTAIN" in metrics:
        print("\n\nEntity Level (MOUNTAIN):")
        print(f"  Precision: {metrics['MOUNTAIN']['precision']:.4f}")
        print(f"  Recall:    {metrics['MOUNTAIN']['recall']:.4f}")
        print(f"  F1:        {metrics['MOUNTAIN']['f1']:.4f}")
    print("-" * 60 + "\n")


def save_metrics_plot(history: dict, output_dir: str):
    """
    Generates and saves a 2x2 grid of plots for Loss, F1, Precision/Recall, and Accuracy.
    """
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(14, 10))

    # Loss Plot
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss', marker='o', color='blue')
    plt.plot(epochs, history['val_loss'], label='Val Loss', marker='o', color='red')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # F1 Score Plot
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['train_f1'], label='Train F1', marker='o', color='blue')
    plt.plot(epochs, history['val_f1'], label='Val F1', marker='o', color='red')
    plt.title('Overall F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Precision & Recall Plot
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history['train_precision'], label='Train Precision', linestyle='--', color='blue')
    plt.plot(epochs, history['val_precision'], label='Val Precision', marker='o', color='blue')
    plt.plot(epochs, history['train_recall'], label='Train Recall', linestyle='--', color='green')
    plt.plot(epochs, history['val_recall'], label='Val Recall', marker='o', color='green')
    plt.title('Precision and Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Accuracy Plot
    plt.subplot(2, 2, 4)
    plt.plot(epochs, history['train_accuracy'], label='Train Accuracy', marker='o', color='blue')
    plt.plot(epochs, history['val_accuracy'], label='Val Accuracy', marker='o', color='red')
    plt.title('Overall Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "training_metrics.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"\n=> Metrics plot successfully saved to: {plot_path}")