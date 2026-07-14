import torch.nn as nn
from transformers import DistilBertModel


class DistilBERTClass(nn.Module):
    def __init__(self, model_name="distilbert-base-cased", num_labels=7):
        super().__init__()

        # Loading raw BERT (produces contextual token embeddings)
        self.encoder = DistilBertModel.from_pretrained(model_name)
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


        # # Manually compute loss if labels are provided (useful during training)
        # loss = None
        # if labels is not None:
        #     loss_fn = nn.CrossEntropyLoss()
        #     # Only keep active parts of the loss (ignore padding or subword dummy tags like -100)
        #     if attention_mask is not None:
        #         active_loss = attention_mask.view(-1) == 1
        #         active_logits = logits.view(-1, self.classifier.out_features)
        #         active_labels = torch.where(
        #             active_loss, labels.view(-1), torch.tensor(loss_fn.ignore_index).type_as(labels)
        #         )
        #         loss = loss_fn(active_logits, active_labels)
        #     else:
        #         loss = loss_fn(logits.view(-1, self.classifier.out_features), labels.view(-1))
        #
        # return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}