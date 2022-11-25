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
        self.linear = nn.Linear(configuration.hidden_size, num_label)

    def forward(self, x):
        outputs, hidden = self.bert(x)
        pred = self.linear(outputs)
        return pred

class TestModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(28*28, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        h1 = nn.functional.relu(self.l1(x))
        h2 = nn.functional.relu(h1)
        do = self.dropout(h2 + h1)
        logits = self.l3(do)
        return logits

