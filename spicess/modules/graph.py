import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import FloatTensor, LongTensor

class GraphConvolution(nn.Module):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim: int, output_dim: int, dropout:int=0., act=F.relu, bias: bool=True):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.act = act
        self.weight = nn.parameter.Parameter(torch.FloatTensor(input_dim, output_dim))
        if bias:
            self.bias = nn.parameter.Parameter(torch.FloatTensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """Create a weight variable with Glorot & Bengio (AISTATS 2010)
        initialization.
        """
        init_range = np.sqrt(6.0 / (self.input_dim + self.output_dim))
        nn.init.uniform_(self.weight, a=-init_range, b=init_range)
        
        if self.bias is not None:
            nn.init.uniform_(self.bias,a=-init_range, b=init_range)    
        
    def forward(self, input: FloatTensor, adj: LongTensor) -> FloatTensor:
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        output = self.act(output)
        return output

    def __repr__(self) -> str:
        return self.__class__.__name__ + ' (' + str(self.input_dim) + ' -> '+ str(self.output_dim) + ')'