import torch.nn.functional as F
import torch.nn as  nn
import torch

class InnerProductDecoder(nn.Module):
    """Decoder model layer for link prediction."""
    def __init__(self, dropout, act=lambda x: x):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, inputs):
        inputs=F.dropout(inputs, self.dropout, training=self.training)
        x=torch.mm(inputs, inputs.t())
        outputs = self.act(x)
        return outputs
