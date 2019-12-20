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

        print(f"X shape before: {x.shape}")
        print(f"H shape before: {h.shape}")

        x = x.view(1, -1, x.size(2))    # x.size(2): embedding dim
        h = h.view(1, -1, h.size(1))    # h.size(1): feature map size (multiplied spatials)

        print(f"X shape: {x.shape}")
        print(f"H shape: {h.shape}")

        prediction, (h, c) = self.lstm(x, (h, c))
        prediction = prediction.view(-1, prediction.size(2))
        prediction = self.linear_out(prediction)
        prediction = self.log_softmax(prediction)

        return prediction, (h, c)

    def init_hidden(self):
        return torch.autograd.Variable(torch.zeros(1, 1, self.lstm_hidden_size), requires_grad=True)
