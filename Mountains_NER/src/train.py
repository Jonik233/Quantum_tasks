import os
import torch
import evaluate
import argparse
from tqdm import tqdm
import torch.nn as nn
from config import label_names
from model import DistilBERTClass
from data_utils import load_dataloaders
from transformers import DistilBertTokenizerFast
from evaluation import compute_metrics, print_metrics


def train(epoch, model, device, train_dataloader, optimizer, loss_function, metrics_evaluator):
    model.train()
    total_train_loss = 0

    # Used during epoch evaluation
    epoch_logits = list()
    epoch_labels = list()

    for batch in tqdm(train_dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        # Returns logits of shape: batch_size, seq_len, num_labels
        logits = model(input_ids=input_ids, attention_mask=attention_mask)

        # Flatten logits and labels to compute loss
        active_logits = logits.view(-1, (len(label_names)))
        active_labels = labels.view(-1)

        loss = loss_function(active_logits, active_labels)

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

        epoch_logits.append(logits.detach().cpu().numpy())
        epoch_labels.append(labels.detach().cpu().numpy())

    epoch_loss = total_train_loss / len(train_dataloader)
    epoch_metrics = compute_metrics(metrics_evaluator, epoch_logits, epoch_labels)

    print(f"\nEpoch {epoch+1}: (Training phase)\n" + "-"*60)
    print(f"Train Loss: {epoch_loss:.4f}")
    print_metrics(epoch_metrics, phase="Training")


@torch.no_grad()
def validate(epoch, model, device, val_dataloader, loss_function, metrics_evaluator):
    model.eval()
    total_val_loss = 0

    # Used during epoch evaluation
    epoch_logits = list()
    epoch_labels = list()

    for batch in tqdm(val_dataloader, desc="Validation"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)

        active_logits = logits.view(-1, len(label_names))
        active_labels = labels.view(-1)

        loss = loss_function(active_logits, active_labels)
        total_val_loss += loss.item()

        epoch_logits.append(logits.detach().cpu().numpy())
        epoch_labels.append(labels.detach().cpu().numpy())

    epoch_loss = total_val_loss / len(val_dataloader)
    epoch_metrics = compute_metrics(metrics_evaluator, epoch_logits, epoch_labels)

    print(f"\nEpoch {epoch+1}: (Validation phase)\n" + "-"*60)
    print(f"Val Loss: {epoch_loss:.4f}")
    print_metrics(epoch_metrics, phase="Validation")

    return epoch_loss


def main():
    print("\n======== TRAINING STARTED ========")

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--valid_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)

    # SageMaker injects these environment variables automatically
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--valid", type=str, default=os.environ.get("SM_CHANNEL_VALID"))
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))

    args = parser.parse_args()
    EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    TRAIN_BATCH_SIZE = args.train_batch_size
    VAL_BATCH_SIZE = args.valid_batch_size

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-cased", use_fast=True)

    train_parameters = {
        "batch_size": TRAIN_BATCH_SIZE,
        "shuffle": True,
        "num_workers": 0
    }

    val_parameters = {
        "batch_size": VAL_BATCH_SIZE,
        "shuffle": True,
        "num_workers": 0
    }

    train_dataloader, val_dataloader = load_dataloaders(tokenizer=tokenizer,
                                                        train_parameters=train_parameters,
                                                        val_parameters=val_parameters,
                                                        train_data_path=args.train,
                                                        val_data_path=args.valid)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DistilBERTClass()
    model.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss(ignore_index=-100)

    evaluator = evaluate.load("seqeval")
    best_val_loss = float('inf')
    output_dir = args.model_dir
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(EPOCHS):
        print(f"\n======== Starting epoch: {epoch+1} ========")
        train(epoch, model, device, train_dataloader, optimizer, loss_function, evaluator)
        val_loss = validate(epoch, model, device, val_dataloader, loss_function, evaluator)

        # Model checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.bin"))
            tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()