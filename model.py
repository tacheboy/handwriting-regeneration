import torch.nn as nn

class HandwritingLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=256, num_layers=3, output_size=3, dropout=0.3):
        super(HandwritingLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Multi-layer LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        # self.ReLU = nn.ReLU()
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden=None):
        # x shape: (batch, seq_len, input_size)
        out, hidden = self.lstm(x, hidden) if hidden is not None else self.lstm(x)
        # out shape: (batch, seq_len, hidden_size)
        output = self.fc(out)
        return output, hidden
