import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim1=64, hidden_dim2=32, dropout=0.2):
        super(GRUModel, self).__init__()
        
        self.gru1 = nn.GRU(input_size=input_dim, hidden_size=hidden_dim1, batch_first=True)
        self.gru2 = nn.GRU(input_size=hidden_dim1, hidden_size=hidden_dim2, batch_first=True)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim2, 1)
        
    def forward(self, x):
        out, _ = self.gru1(x)
        out, h_n = self.gru2(out)
        
        # h_n from second GRU: (1, batch, hidden_dim2)
        last_hidden = h_n.squeeze(0)
        
        out = self.dropout(last_hidden)
        out = self.fc(out)
        return out
