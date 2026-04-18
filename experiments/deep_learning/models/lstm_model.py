import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim1=64, hidden_dim2=32, dropout=0.2):
        super(LSTMModel, self).__init__()
        
        # LSTM Layers
        self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hidden_dim1, hidden_size=hidden_dim2, batch_first=True)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim2, 1)
        
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        
        # LSTM outputs: (output, (h_n, c_n))
        # We only need the last hidden state for forecasting
        out, _ = self.lstm1(x)
        out, (h_n, _) = self.lstm2(out)
        
        # h_n from the second LSTM has shape (1, batch, hidden_dim2)
        # Squeeze to (batch, hidden_dim2)
        last_hidden = h_n.squeeze(0)
        
        out = self.dropout(last_hidden)
        out = self.fc(out)
        return out
