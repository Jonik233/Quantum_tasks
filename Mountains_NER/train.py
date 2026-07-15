import os
import torch
import argparse
from tqdm import tqdm
import torch.nn as nn
from config import label_names
from model import DistilBERTClass
from data_utils import load_dataloaders
from transformers import DistilBertTokenizerFast


def train(epoch, model, device, train_dataloader, optimizer, loss_function):
    model.train()
    total_train_loss = 0

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

    epoch_loss = total_train_loss / len(train_dataloader)
    print(f"Epoch {epoch+1} Train Loss: {epoch_loss:.4f}")


@torch.no_grad()
def validate(epoch, model, device, val_dataloader, loss_function):
    model.eval()
    total_val_loss = 0

    for batch in tqdm(val_dataloader, desc="Validation"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)

        active_logits = logits.view(-1, len(label_names))
        active_labels = labels.view(-1)

        loss = loss_function(active_logits, active_labels)
        total_val_loss += loss.item()

    epoch_loss = total_val_loss / len(val_dataloader)
    print(f"Epoch {epoch+1} Val Loss: {epoch_loss:.4f}")

    return epoch_loss


def main():
    print("TRAINING STARTED")

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--valid_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-cased",
                                                        use_fast=True,
                                                        cache_dir="./tokenizer")

    train_dataloader, val_dataloader = load_dataloaders(tokenizer)

    model = DistilBERTClass()
    model.to(device)

    LEARNING_RATE = args.learning_rate
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    loss_function = nn.CrossEntropyLoss(ignore_index=-100)

    EPOCHS = args.epochs

    best_val_loss = float('inf')
    output_dir = os.environ["SAVE_MODEL_DIR"]
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(EPOCHS):
        print(f"Starting epoch: {epoch+1}")
        train(epoch, model, device, train_dataloader, optimizer, loss_function)
        val_loss = validate(epoch, model, device, val_dataloader, loss_function)

        # Model checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.bin"))
            tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()