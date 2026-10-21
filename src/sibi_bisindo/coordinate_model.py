"""Small coordinate CNN, optional MLP control; no image pixels are inputs."""
import torch
from torch import nn


class CoordinateNet(nn.Module):
    def __init__(self, classes, kind='cnn'):
        super().__init__()
        self.kind = kind
        if kind == 'cnn':
            self.encoder = nn.Sequential(nn.Conv1d(7, 32, 3, padding=1), nn.ReLU(),
                                         nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(),
                                         nn.MaxPool1d(2), nn.Flatten(), nn.Linear(64*21, 64), nn.ReLU())
        elif kind == 'mlp':
            self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(42*7, 128), nn.ReLU(),
                                         nn.Linear(128, 64), nn.ReLU())
        else:
            raise ValueError('Model kind must be cnn or mlp')
        self.dropout = nn.Dropout(.2)
        self.classifier = nn.Linear(64, classes)

    def embed(self, x):
        return self.encoder(x.transpose(1, 2).contiguous() if self.kind == 'cnn' else x)

    def forward(self, x):
        return self.classifier(self.dropout(self.embed(x)))
