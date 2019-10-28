import torch.optim as optim
import torch.nn as nn
from torch.utils import data
from data import ANetCaptionsDataset
from model import DecoderLSTM


def run_single_epoch(data_loader, model, optimizer, criterion):
    """
    Run the model for a single epoch.
    :param data_loader: Customized PyTorch inherited data loader for the dataset.
    :param model: Model to train.
    :param optimizer: Optimization technique.
    :param criterion: Loss function to use.
    :return: Epoch summary.
    """
    loss = 0.0

    for x, target_description in data_loader:
        x = x[2]
        optimizer.zero_grad()

        target_description_length = 20

        decoder_input = 0 # SOS Token ???

        # Teacher forced
        for idx in range(target_description_length):
            pred, h = model(decoder_input)
            loss += criterion(pred, target_description[idx])
            decoder_input = target_description[idx] + x

        # Backprop after every batch
        loss.backward()
        optimizer.step()

    return loss


if __name__ == '__main__':
    anet_path = '../gvd-data/ActivityNet/data/anet/anet_annotations_trainval.json'
    features_path = '../gvd-data/ActivityNet/data'

    train_anet = ANetCaptionsDataset(anet_path, features_path, train=True)
    validation_anet = ANetCaptionsDataset(anet_path, features_path, train=False)
    print("Training size: {}, Validation Size: {}".format(len(train_anet), len(validation_anet)))

    train_anet_generator = data.DataLoader(train_anet, batch_size=32, num_workers=8)
    validation_anet_generator = data.DataLoader(validation_anet)

    num_epochs = 25
    learning_rate = 0.00001
    visual_feature_size = 1024
    lstm_hidden_size = 256
    vocab_size = 10000
    net = DecoderLSTM(visual_feature_size, lstm_hidden_size, vocab_size)
    opt = optim.Adam(params=net.parameters(), lr=learning_rate)
    loss = nn.NLLLoss()

    # for epoch in num_epochs:
    #     epoch_summary = run_single_epoch(data_loader=train_anet, model=net, optimizer=opt, criterion=loss)

    for x, label in train_anet_generator:
        print(label[0])
        # print("Vid Duration: {}".format(x[:, 1]))
        # print("Feature Map Size: {}".format(x[:, 2].shape))
        # print("(Start, End): ({}, {})".format(x[:, 3], x[:, 4]))
        # print("Description: {}".format(label))
