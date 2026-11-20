import torch
import torch.nn as nn

from model_batchnorm import CNN


class ResidualConv1dBlock(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        padding = dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.net(x))


class PlateRecognizerSingleInputV2(nn.Module):
    def __init__(self, vocab_size=70, seq_len=8):
        super().__init__()
        self.cnn = CNN()
        self.seq_len = seq_len
        self.proj = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.sequence = nn.Sequential(
            ResidualConv1dBlock(256, dilation=1),
            ResidualConv1dBlock(256, dilation=2),
            ResidualConv1dBlock(256, dilation=4),
            ResidualConv1dBlock(256, dilation=1),
        )
        self.classifier = nn.Conv1d(256, vocab_size, kernel_size=1)

    def forward(self, image):
        x = self.cnn.relu1(self.cnn.bn1(self.cnn.conv1(image)))
        x = self.cnn.relu2(self.cnn.bn2(self.cnn.conv2(x)))
        x = self.cnn.relu3(self.cnn.bn3(self.cnn.conv3(x)))
        x = x.mean(dim=2)
        x = nn.functional.adaptive_avg_pool1d(x, self.seq_len)
        x = self.proj(x)
        x = self.sequence(x)
        x = self.classifier(x)
        return x.transpose(1, 2)


def load_cnn_from_checkpoint(model, checkpoint):
    model.cnn.load_state_dict(checkpoint["cnn"])
    return model
