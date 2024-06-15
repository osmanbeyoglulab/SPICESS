import torch.nn.functional as F
import torch.nn as  nn

class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0, input_dropout=0):
        super(LinearBlock, self).__init__()
        self.input_dropout = input_dropout
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.block = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                nn.Linear(output_dim, 2*output_dim),
                nn.BatchNorm1d(2*output_dim),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                nn.Linear(2*output_dim, output_dim),
            )
        
        # for m in self.block:
        #     self.weights_init(m)
    
    def forward(self, x, A=None):
        x = F.dropout(x, self.input_dropout, self.training)
        return self.block(x)
    
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)      
                
    def __repr__(self) -> str:
        return self.__class__.__name__ + ' (' + str(self.input_dim) + ' -> ' + str(self.output_dim) + ')'
        
    



