import torch
import torch.nn as nn

class CNNLSTMModel(nn.Module):
    def __init__(self, input_dim, conv_filters=32, lstm_units=64):
        super(CNNLSTMModel, self).__init__()
        
        # Conv1D expects (batch, channels, length)
        # Our input is (batch, length, features)
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=conv_filters, kernel_size=3)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv1d(in_channels=conv_filters, out_channels=conv_filters, kernel_size=3)
        self.relu2 = nn.ReLU()
        
        self.lstm = nn.LSTM(conv_filters, lstm_units, batch_first=True)
        
        self.fc = nn.Linear(lstm_units, 1)

    def forward(self, x):
        # Permute to (batch, features, length) for Conv1d
        x = x.permute(0, 2, 1)
        
        out = self.conv1(x)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.relu2(out)
        
        # Permute back to (batch, length, features) for LSTM
        out = out.permute(0, 2, 1)
        
        out, _ = self.lstm(out)
        
        # Take the output of the last time step
        out = out[:, -1, :]
        out = self.fc(out)
        return out.squeeze()
