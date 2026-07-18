from config import label_names
from logging_utils import logger


def compute_metrics(evaluator, predictions, labels):
    """
    Computes seqeval metrics for a given window of predictions.
    """
    mapped_predictions = []
    mapped_labels = []

    # Iterate through each batch in the epoch
    for batch_predictions, batch_labels in zip(predictions, labels):

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


def log_metrics(loss, metrics: dict, phase: str, step: int):
    """
    Logs the calculated metrics to the terminal for quick checks.
    """
    logger.info(f"------ {phase.upper()} METRICS (Epoch {step + 1}) ------ ")
    logger.info(f"Loss: {loss:.4f}")
    logger.info(f"Overall Precision: {metrics.get('overall_precision', 0):.4f}")
    logger.info(f"Overall Recall:    {metrics.get('overall_recall', 0):.4f}")
    logger.info(f"Overall F1:        {metrics.get('overall_f1', 0):.4f}")
    logger.info(f"Overall Accuracy:  {metrics.get('overall_accuracy', 0):.4f}")

    if "MOUNTAIN" in metrics:
        logger.info("Entity Level (MOUNTAIN):")
        logger.info(f"  Precision: {metrics['MOUNTAIN']['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['MOUNTAIN']['recall']:.4f}")
        logger.info(f"  F1:        {metrics['MOUNTAIN']['f1']:.4f}")
    logger.info("-" * 60)