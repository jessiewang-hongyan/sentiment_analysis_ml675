import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, BertConfig
import pytorch_lightning as pl


class BertClassifier(BertPreTrainedModel):
    def __init__(self, num_label):
        configuration = BertConfig()
        super(BertClassifier, self).__init__(configuration)

        self.num_label = num_label
        self.bert = BertModel(configuration)
        self.linear = LinearClassifier(configuration.hidden_size, self.num_label, dropout_ratio=0.2)

    def forward(self, x):
        logits = x['logits']
        mask = x['mask']
        outputs = self.bert(logits, encoder_attention_mask = mask).last_hidden_state
        pred = self.linear(outputs)
        return pred[:, -1, :]

    def predict(self, x):
        logits = x['logits']
        mask = x['mask']
        pred = self(logits, encoder_attention_mask = mask).last_hidden_state
        return torch.argmax(pred, dim=1)

class LinearClassifier(nn.Module):
    def __init__(self, in_size, out_size, dropout_ratio=0.8):
        super().__init__()
        
        self.in_size = in_size
        self.out_size = out_size

        self.linear = nn.Linear(in_features=in_size, out_features=out_size)
        self.drop = nn.Dropout(dropout_ratio)
        self.softmax = nn.Softmax()

    def forward(self, x):
        outputs = self.linear(x)
        outputs = self.drop(outputs)
        return self.softmax(outputs)

# class TestModel(pl.LightningModule):
#     def __init__(self):
#         super().__init__()
#         self.l1 = nn.Linear(28*28, 64)
#         self.l2 = nn.Linear(64, 64)
#         self.l3 = nn.Linear(64, 10)
#         self.dropout = nn.Dropout(0.1)

#     def forward(self, x):
#         h1 = nn.functional.relu(self.l1(x))
#         h2 = nn.functional.relu(h1)
#         do = self.dropout(h2 + h1)
#         logits = self.l3(do)
#         return logits

