import torch
import torch.nn as nn


class DecoderLSTM(nn.Module):
    def __init__(self, visual_feature_size, wv_size, vocab_size):
        super().__init__()
        self.linear_in = nn.Linear(visual_feature_size, wv_size)
        self.lstm = nn.LSTM(wv_size, wv_size)
        self.linear_out = nn.Linear(wv_size, vocab_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        pred = self.linear_in(x)
        pred, (h, c) = self.lstm(pred)
        pred = self.softmax(pred)

        return pred, h
