import torch
import torch.nn as nn

class GRUResidualModel(nn.Module):
    def __init__(self, input_dim=45, hidden_dim_1=64, hidden_dim_2=32, dropout=0.2):
        super(GRUResidualModel, self).__init__()
        
        # First GRU layer
        self.gru1 = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim_1,
            batch_first=True
        )
        
        # Normalization layer for stability
        self.layer_norm = nn.LayerNorm(hidden_dim_1)
        
        # Second GRU layer
        self.gru2 = nn.GRU(
            input_size=hidden_dim_1,
            hidden_size=hidden_dim_2,
            batch_first=True
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim_2, 1)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        
        # First GRU pass
        # out1 shape: (batch, seq_len, hidden_dim_1)
        out1, _ = self.gru1(x)
        
        # Apply normalization
        out1 = self.layer_norm(out1)
        
        # Second GRU pass
        # out2 shape: (batch, seq_len, hidden_dim_2)
        out2, _ = self.gru2(out1)
        
        # Take the last time step's output for many-to-one mapping
        # last_out shape: (batch, hidden_dim_2)
        last_out = out2[:, -1, :]
        
        # Apply dropout
        last_out = self.dropout(last_out)
        
        # Final prediction
        # prediction shape: (batch, 1)
        prediction = self.fc(last_out)
        
        return prediction

if __name__ == "__main__":
    # Smoke test
    model = GRUResidualModel()
    dummy_input = torch.randn(8, 192, 45)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (8, 1)
    print("GRUResidualModel smoke test passed!")
