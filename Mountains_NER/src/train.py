import os
import torch
import mlflow
import evaluate
import argparse
from tqdm import tqdm
import torch.nn as nn
from logging_utils import logger
from config import label_names
from model import DistilBERTClass
from data_utils import load_dataloaders
from transformers import DistilBertTokenizerFast
from evaluation import compute_metrics, log_metrics


def train(epoch, model, device, train_dataloader, optimizer, loss_function,
          metrics_evaluator, global_step, log_every_n_batches=20):
    model.train()

    epoch_predictions = list()
    epoch_labels = list()
    tracked_loss = 0.0
    window_loss = 0.0

    progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")

    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask)

        active_logits = logits.view(-1, len(label_names))
        active_labels = labels.view(-1)
        loss = loss_function(active_logits, active_labels)

        loss.backward()
        optimizer.step()

        global_step += 1
        tracked_loss += loss.item()
        window_loss += loss.item()

        # Streaming Training Loss to MLflow and updating terminal
        if global_step % log_every_n_batches == 0:
            avg_window_loss = window_loss / log_every_n_batches
            mlflow.log_metric("train_loss_step", avg_window_loss, step=global_step)
            progress_bar.set_postfix({"Loss": f"{avg_window_loss:.4f}"})
            window_loss = 0.0

        predictions = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        epoch_predictions.append(predictions)
        epoch_labels.append(labels.detach().cpu().numpy())

    # Calculating epoch-wise train metrics
    epoch_train_loss = tracked_loss / len(train_dataloader)
    train_metrics = compute_metrics(metrics_evaluator, epoch_predictions, epoch_labels)

    # Loging epoch-wise metrics to MLflow
    log_metrics(epoch_train_loss, train_metrics, phase="Validation", step=epoch)
    mlflow.log_metric("train_f1_epoch", train_metrics.get('overall_f1', 0), step=epoch)
    mlflow.log_metric("train_precision_epoch", train_metrics.get('overall_precision', 0), step=epoch)
    mlflow.log_metric("train_recall_epoch", train_metrics.get('overall_recall', 0), step=epoch)

    return global_step


@torch.no_grad()
def validate(epoch, model, device, val_dataloader, loss_function, metrics_evaluator):
    model.eval()

    val_predictions = []
    val_labels = []
    tracked_loss = 0

    for batch in tqdm(val_dataloader, desc=f"Validation Epoch {epoch + 1}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        active_logits = logits.view(-1, len(label_names))
        active_labels = labels.view(-1)

        loss = loss_function(active_logits, active_labels)
        tracked_loss += loss.item()

        predictions = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        val_predictions.append(predictions)
        val_labels.append(labels.detach().cpu().numpy())

    avg_val_loss = tracked_loss / len(val_dataloader)
    val_metrics = compute_metrics(metrics_evaluator, val_predictions, val_labels)

    # Log to MLflow and Console
    log_metrics(avg_val_loss, val_metrics, phase="Validation", step=epoch)
    mlflow.log_metric("val_loss_epoch", avg_val_loss, step=epoch)
    mlflow.log_metric("val_f1_epoch", val_metrics.get('overall_f1', 0), step=epoch)
    mlflow.log_metric("val_precision_epoch", val_metrics.get('overall_precision', 0), step=epoch)
    mlflow.log_metric("val_recall_epoch", val_metrics.get('overall_recall', 0), step=epoch)

    return avg_val_loss


def main():
    logger.info("================ TRAINING STARTED ================")

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--valid_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--log_every_n_batches", type=int, default=20)

    parser.add_argument("--mlflow_tracking_arn", type=str, required=True)
    parser.add_argument("--mlflow_experiment_name", type=str, default="Mountains_NER_Experiment")

    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--valid", type=str, default=os.environ.get("SM_CHANNEL_VALID"))
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))

    args = parser.parse_args()

    mlflow.set_tracking_uri(args.mlflow_tracking_arn)
    mlflow.set_experiment(args.mlflow_experiment_name)

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-cased", use_fast=True)

    train_dataloader, val_dataloader = load_dataloaders(
        tokenizer=tokenizer,
        train_parameters={"batch_size": args.train_batch_size, "shuffle": True, "num_workers": 0},
        val_parameters={"batch_size": args.valid_batch_size, "shuffle": True, "num_workers": 0},
        train_data_path=args.train,
        val_data_path=args.valid
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DistilBERTClass().to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)
    loss_function = nn.CrossEntropyLoss(ignore_index=-100)
    evaluator = evaluate.load("seqeval")

    output_dir = args.model_dir
    os.makedirs(output_dir, exist_ok=True)
    best_val_loss = float('inf')
    global_step = 0

    # Initialize MLflow Run
    with mlflow.start_run():
        mlflow.log_params({
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "train_batch_size": args.train_batch_size
        })

        for epoch in range(args.epochs):
            logger.info(f"================ Starting epoch: {epoch + 1} ================")

            global_step = train(epoch, model, device, train_dataloader, optimizer,
                                loss_function, evaluator, global_step, args.log_every_n_batches)

            val_loss = validate(epoch, model, device, val_dataloader, loss_function, evaluator)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(output_dir, "best_model.bin"))
                tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()