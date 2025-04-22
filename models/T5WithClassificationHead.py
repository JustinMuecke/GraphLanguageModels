import torch.nn as nn
import torch
from transformers import T5ForConditionalGeneration

class T5WithClassificationHead(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.classifier = torch.nn.Linear(self.model.config.d_model, 1)  # Binary classification head

    def forward(self, input_ids, attention_mask, labels=None):
        # Forward pass through the T5 encoder
        encoder_outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Get the representation of the last token (or pool the output in some way)
        last_hidden_state = encoder_outputs.last_hidden_state
        pooled_output = last_hidden_state.mean(dim=1)  # Use the last token's hidden state as sequence representation

        # Apply classification head
        logits = self.classifier(pooled_output)

        # If labels are provided, compute the loss
        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.float())
            return loss, logits
        return logits
