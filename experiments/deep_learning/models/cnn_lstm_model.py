import torch
import torch.nn as nn

class CNNLSTMModel(nn.Module):
    def __init__(self, input_dim, cnn_filters=32, lstm_hidden=64, dropout=0.2):
        super(CNNLSTMModel, self).__init__()
        
        # Conv1D expects (batch, channels, length)
        # Our input is (batch, length, features) -> will need permutation
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=cnn_filters, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=cnn_filters, out_channels=cnn_filters, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        
        # After Conv, we permute back to (batch, length, filters) for LSTM
        self.lstm = nn.LSTM(input_size=cnn_filters, hidden_size=lstm_hidden, batch_first=True)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden, 1)
        
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        
        # Conv1D path
        x = x.permute(0, 2, 1) # (batch, channels, length)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        
        # LSTM path
        x = x.permute(0, 2, 1) # (batch, length, channels)
        out, (h_n, _) = self.lstm(x)
        
        # Last hidden state
        last_hidden = h_n.squeeze(0)
        
        out = self.dropout(last_hidden)
        out = self.fc(out)
        return out
