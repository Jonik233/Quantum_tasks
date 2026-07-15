import os
import torch.nn as nn
from dotenv import load_dotenv
from transformers import DistilBertModel

load_dotenv()


class DistilBERTClass(nn.Module):
    def __init__(self, model_name="distilbert-base-cased", num_labels=3):
        super().__init__()

        # Loading raw BERT (produces contextual token embeddings)
        self.encoder = DistilBertModel.from_pretrained(model_name, cache_dir=os.environ["BERT_CACHE_DIR"])
        hidden_size = self.encoder.config.hidden_size

        # Model head (produces logits)
        self.dropout = nn.Dropout(p=0.1)
        self.pre_classifier = nn.Linear(hidden_size, hidden_size // 2)
        self.activation = nn.GELU()
        self.classifier = nn.Linear(hidden_size // 2, num_labels)


    def forward(self, input_ids, attention_mask=None):

        # Produce hidden states (contextual embeddings)
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Extract hidden state (batch_size, sequence_length, hidden_size)
        hidden_state = outputs.last_hidden_state

        # Applying classification head
        hidden_state = self.dropout(hidden_state)
        activation_scores = self.activation(self.pre_classifier(hidden_state))
        logits = self.classifier(activation_scores)  # Output shape: (batch_size, seq_len, num_labels)

        return logits