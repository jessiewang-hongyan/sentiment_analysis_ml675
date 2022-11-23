import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, BertConfig

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

