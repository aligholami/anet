import torch
import torch.nn as nn


class DecoderLSTM(nn.Module):
    def __init__(self, visual_feature_size, lstm_hidden_size, vocab_size):
        super().__init__()
        self.linear_in = nn.Linear(visual_feature_size, lstm_hidden_size)
        self.lstm = nn.LSTM(lstm_hidden_size, lstm_hidden_size)
        self.linear_out = nn.Linear(lstm_hidden_size, vocab_size)
        self.lsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        pred = self.linear_in(x)
        pred = pred.view(-1, 1, pred.size(1))
        pred, (h, c) = self.lstm(pred)
        pred = self.linear_out(pred)
        pred = self.lsoftmax(pred)

        return pred, h
