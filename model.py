import torch
import torch.nn as nn


class DecoderLSTM(nn.Module):
    def __init__(self, visual_feature_size, lstm_hidden_size, vocab_size):
        super().__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self.visual_feature_size = visual_feature_size
        self.linear_in = nn.Linear(visual_feature_size, lstm_hidden_size)
        self.embedding = nn.Embedding(vocab_size, lstm_hidden_size)
        self.lstm = nn.LSTM(lstm_hidden_size, lstm_hidden_size)
        self.linear_out = nn.Linear(lstm_hidden_size, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, h, c):
        x = self.embedding(x.long())
        h = self.linear_in(h.float())

        x = x.view(-1, 1, x.size(1))
        h = h.view(1, -1, h.size(1))

        pred, (h, c) = self.lstm(x, (h, c))
        pred = pred.view(-1, pred.size(2))
        pred = self.linear_out(pred)
        pred = self.log_softmax(pred)

        return pred, (h, c)

    def init_hidden(self):
        return torch.autograd.Variable(torch.zeros(1, 1, self.lstm_hidden_size), requires_grad=True)
