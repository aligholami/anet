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
        decoder_input = x.view(-1, x.shape[1] * x.shape[2])

        # Teacher forced
        for idx in range(len(target_description)):
            pred, h = model(decoder_input.float())
            print("Prediction: {}".format(pred))
            print("Target Description: {}".format(target_description[idx]))
            loss += criterion(pred, target_description[idx])

        # Backprop after every batch
        loss.backward()
        optimizer.step()

    return loss


if __name__ == '__main__':
    anet_path = '../gvd-data/ActivityNet/data/anet/anet_annotations_trainval.json'
    features_path = '../gvd-data/ActivityNet/data'

    train_anet = ANetCaptionsDataset(anet_path, features_path, train=True)
    validation_anet = ANetCaptionsDataset(anet_path, features_path, train=False)
    # print("Training size: {}, Validation Size: {}".format(len(train_anet), len(validation_anet)))

    train_anet_generator = data.DataLoader(train_anet, batch_size=1)
    validation_anet_generator = data.DataLoader(validation_anet)

    num_epochs = 25
    learning_rate = 0.00001
    visual_feature_size = train_anet.get_max_fm_size() * 1024
    lstm_hidden_size = 256
    vocab_size = train_anet.get_vocab_size()
    net = DecoderLSTM(visual_feature_size, lstm_hidden_size, vocab_size)
    opt = optim.Adam(params=net.parameters(), lr=learning_rate)
    loss = nn.NLLLoss()

    print("Visual Feature Size: {}".format(visual_feature_size))
    print("Vocab Size: {}".format(vocab_size))

    for epoch in range(num_epochs):
        epoch_summary = run_single_epoch(data_loader=train_anet_generator, model=net, optimizer=opt, criterion=loss)