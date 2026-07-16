from config import label_names


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
    print(f"\tLoss: {loss:.4f}")
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