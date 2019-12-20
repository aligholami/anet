import torch
import torch.nn as nn
import torch.nn.functional as F


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
        x = x.view(1, -1, x.size(2))    # x.size(2): embedding dim
        h = h.view(1, -1, h.size(1))    # h.size(1): feature map size (multiplied spatials)
        prediction, (h, c) = self.lstm(x, (h, c))
        prediction = prediction.view(-1, prediction.size(2))
        prediction = self.linear_out(prediction)
        prediction = self.log_softmax(prediction)

        return prediction, (h, c)

    def init_cell(self, batch_size):
        return torch.autograd.Variable(torch.zeros(1, batch_size, self.lstm_hidden_size), requires_grad=True)

    def init_hidden(self, batch_size, vf):
        hidden_init_tensor = torch.autograd.Variable(torch.zeros(1, batch_size, self.visual_feature_size), requires_grad=True)
        hidden_init_tensor[:, :, :] = vf.view(1, batch_size, self.visual_feature_size)
        hidden_init_tensor = self.linear_in(hidden_init_tensor)
        return hidden_init_tensor
