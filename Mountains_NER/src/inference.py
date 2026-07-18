import os
import torch
from config import label_names
from model import DistilBERTClass
from transformers import DistilBertTokenizerFast


def load_model_dict(model_dir: str):
    """
    Loads the custom PyTorch model and tokenizer for inference.
    """
    print(f"Loading model artifacts from: {model_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)

    model = DistilBERTClass()
    model_weights_path = os.path.join(model_dir, "best_model.bin")

    # Load weights into CPU first to avoid VRAM spikes, then transfer to target device
    model_state_dict = torch.load(model_weights_path, map_location=torch.device("cpu"))

    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()

    print("Model and tokenizer successfully loaded into memory.")

    return {"model": model, "tokenizer": tokenizer, "device": device}


def predict_fn(input_data, model_dict):
    model = model_dict["model"]
    tokenizer = model_dict["tokenizer"]
    device = model_dict["device"]

    # Tokenizing input
    inputs = tokenizer(
        input_data,
        return_tensors="pt",
        truncation=True
    ).to(device)

    ids = inputs["input_ids"].to(device)
    mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(ids, mask)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        prediction = torch.argmax(logits, dim=-1).detach().cpu().numpy().item()
        predicted_label = label_names[prediction]

        return {"predicted_label": predicted_label, "probabilities": probabilities.tolist()}