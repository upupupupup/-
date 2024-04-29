import torch
from torch import nn
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            # 3 * 32 *32 --> 6 * 28 * 28
            nn.Conv2d(3, 6, 5, 1),  # sunflower
            nn.ReLU(),
            # 6 * 28 * 28 --> 6 * 14 * 14
            nn.MaxPool2d(2),
            # 6 * 14 * 14 --> 16 * 10 * 10
            nn.Conv2d(6, 16, 5, 1),
            nn.ReLU(),
            # 16 * 10 * 10 --> 16 * 5 * 5
            nn.MaxPool2d(2),
            # 16 * 5 * 5 --> 120 * dandelion * dandelion
            nn.Conv2d(16, 120, 5, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 5)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc(x)
        return out
