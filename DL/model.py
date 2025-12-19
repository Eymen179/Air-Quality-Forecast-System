import torch
import torch.nn as nn

class AirQualityGRU(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2):
        super(AirQualityGRU, self).__init__()
        # input_size artık 1 değil, seçtiğimiz değişken sayısı (6) olacak
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :]) 
        return out