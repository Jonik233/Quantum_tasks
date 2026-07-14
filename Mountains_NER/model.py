import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel


class DistilBERTClass(nn.Module):

    def __init__(self):
        super().__init__()

        # Backbone of our network
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-cased")

        # Used to further adapt the output representation vector from bert to specific task
        self.pre_classifier = nn.Linear(768, 768)

        self.dropout = nn.Dropout(p=0.3)

        # Final linear classifier that transforms (768, 768) matrix into matrix of logits of size (768, 4)
        self.classifier = nn.Linear(768, 4)


    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)

        # Extract the hidden state: shape (batch_size, 512, 768)
        hidden_state = output_1[0]

        # Get the [CLS] token (first token) for the whole batch
        # This reduces it to a 2D Tensor: (Batch, 768)
        pooler = hidden_state[:, 0]

        pooler = self.pre_classifier(pooler)

        pooler = nn.ReLU()(pooler)

        pooler = self.dropout(pooler)

        output = self.classifier(pooler)

        return output