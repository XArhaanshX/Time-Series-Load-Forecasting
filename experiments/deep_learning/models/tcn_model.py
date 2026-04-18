import torch
import torch.nn as nn
import torch.nn.utils.weight_norm as weight_norm

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TCNBlock, self).__init__()
        # Causal Conv1D using padding
        self.conv1 = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1,
                                 self.conv2, self.relu2, self.dropout2)
        
        # Residual connection if input and output channels differ
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, channels, length)
        out = self.net(x)
        # Chrop the extra padding at the end to keep it causal
        # Padding should be (kernel_size-1)*dilation
        res = x if self.downsample is None else self.downsample(x)
        
        # To make it truly causal, we need to slice the output
        # PyTorch Conv1d with padding (k-1)*d results in length + 2*(k-1)*d
        # We only want the first 'length' steps that don't depend on the future
        # Standard TCN implementation typically uses chrop
        out = out[:, :, :x.size(2)] 
        
        return self.relu(out + res)

class TCNModel(nn.Module):
    def __init__(self, input_dim, num_channels=[32, 32, 32, 32], kernel_size=3, dropout=0.2):
        super(TCNModel, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TCNBlock(in_channels, out_channels, kernel_size, stride=1,
                                dilation=dilation_size, padding=(kernel_size-1) * dilation_size, 
                                dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = x.permute(0, 2, 1) # (batch, channels, length)
        y1 = self.network(x)
        
        # We take the last timestep for prediction
        out = self.fc(y1[:, :, -1])
        return out
