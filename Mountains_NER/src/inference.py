import os
import torch
from config import label_names
from model import DistilBERTClass
from transformers import DistilBertTokenizerFast


def load_model(model_dir: str) -> dict:
    """
    Loads the custom PyTorch model and tokenizer into memory.
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

    print("=> Model and tokenizer successfully loaded into memory.")

    return {"model": model, "tokenizer": tokenizer, "device": device}


@torch.no_grad()
def predict_fn(text: str, model_dict: dict) -> dict:
    """
    Takes a raw string, runs NER inference, and returns a clean
    JSON-serializable dictionary mapping words to entity predictions.
    """
    model = model_dict["model"]
    tokenizer = model_dict["tokenizer"]
    device = model_dict["device"]

    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)

    # Logits shape: (batch_size=1, seq_len, num_labels)
    logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

    probabilities = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    predictions = torch.argmax(logits, dim=-1)[0].cpu().numpy()

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    extracted_entities = []

    # Filter out special tokens and format the API response
    for token, pred_idx, prob_array in zip(tokens, predictions, probabilities):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue

        label = label_names[pred_idx]
        confidence = float(prob_array[pred_idx])

        # If it's a subword (starts with ##) and we have existing entities, merge it
        if token.startswith("##") and extracted_entities:
            prev = extracted_entities[-1]
            prev["token"] += token.replace("##", "")
            # Updating confidence: average of the sub-tokens
            prev["confidence"] = round((prev["confidence"] + confidence) / 2, 4)
        else:
            extracted_entities.append({
                "token": token,
                "label": label,
                "confidence": round(confidence, 4)
            })

    return {
        "original_text": text,
        "predictions": extracted_entities
    }