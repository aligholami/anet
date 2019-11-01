import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils import data
from data import ANetCaptionsDataset
from model import DecoderLSTM
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    show_loss_every_n_iterations = 100
    iteration = 0
    running_loss = 0
    total_loss = 0

    for x, target_description in data_loader:
        vf = x[2]
        optimizer.zero_grad()
        decoder_input = vf.view(-1, vf.shape[1] * vf.shape[2])
        x_type = 'vis'

        # Teacher forced decoder training
        for idx in range(len(target_description)):
            preds, h = model(decoder_input, x_type)
            preds = preds.view(-1, preds.shape[2])
            step_loss = criterion(preds, target_description[idx])
            running_loss += step_loss
            total_loss += step_loss
            x_type = 'lan'
            decoder_input = target_description[idx]

            iteration += 1
            if iteration % show_loss_every_n_iterations == (show_loss_every_n_iterations - 1):
                print(running_loss.item()/show_loss_every_n_iterations)
                running_loss = 0

    total_loss.backward()
    optimizer.step()

    return total_loss


if __name__ == '__main__':
    anet_path = '../gvd-data/ActivityNet/data/anet/anet_annotations_trainval.json'
    features_path = '../gvd-data/ActivityNet/data'
    train_anet = ANetCaptionsDataset(anet_path, features_path, train=True)
    validation_anet = ANetCaptionsDataset(anet_path, features_path, train=False)
    # print("Training size: {}, Validation Size: {}".format(len(train_anet), len(validation_anet)))

    train_anet_generator = data.DataLoader(train_anet, batch_size=32)
    validation_anet_generator = data.DataLoader(validation_anet)

    num_epochs = 25
    learning_rate = 0.01
    visual_feature_size = train_anet.get_max_fm_size() * 1024
    lstm_hidden_size = 256
    vocab_size = train_anet.get_vocab_size()
    net = DecoderLSTM(visual_feature_size, lstm_hidden_size, vocab_size)
    opt = optim.Adam(params=net.parameters(), lr=learning_rate)
    loss = nn.NLLLoss()

    print("Visual Feature Size: {}".format(visual_feature_size))
    print("Vocab Size: {}".format(vocab_size))

    for epoch in range(num_epochs):
        print("Started Epoch {}".format(epoch))
        epoch_summary = run_single_epoch(data_loader=train_anet_generator, model=net, optimizer=opt, criterion=loss)
        print("Epoch Loss: {}".format(epoch_summary))
