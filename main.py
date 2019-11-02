import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils import data
from data import ANetCaptionsDataset
from model import DecoderLSTM
from tqdm import tqdm

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")


def run_single_epoch(data_loader, model, optimizer, criterion, batch_size):
    """
    Run the model for a single epoch.
    :param data_loader: Customized PyTorch inherited data loader for the dataset.
    :param model: Model to train.
    :param optimizer: Optimization technique.
    :param criterion: Loss function to use.
    :return: Epoch summary.
    """
    total_loss = 0.0
    show_loss_every_n_iterations = 100
    iteration = 0
    running_loss = 0
    optimizer.zero_grad()

    for x, target_description in tqdm(data_loader):
        iter_loss = 0
        vf = x[2]
        optimizer.zero_grad()
        decoder_input = vf.view(-1, vf.shape[1] * vf.shape[2])
        decoder_h = model.init_hidden()
        decoder_c = model.init_hidden()
        x_type = 'vis'

        # Teacher forced decoder training
        for idx in range(len(target_description)):
            predictions, (decoder_h, decoder_c) = model(decoder_input.to(device), decoder_h.to(device), decoder_c.to(device), x_type)
            iter_loss += criterion(predictions, target_description[idx].to(device))

        running_loss += iter_loss
        x_type = 'lan'
        decoder_input = target_description[idx]

        iteration += 1
        if iteration % show_loss_every_n_iterations == (show_loss_every_n_iterations - 1):
            print("Loss at step {}: {}".format(iteration, running_loss.item()/show_loss_every_n_iterations))
            running_loss = 0

        iter_loss.backward()
        optimizer.step()

        total_loss += iter_loss

    return total_loss/len(data_loader)


if __name__ == '__main__':
    anet_path = '../gvd-data/ActivityNet/data/anet/anet_annotations_trainval.json'
    features_path = '../gvd-data/ActivityNet/data'
    train_anet = ANetCaptionsDataset(anet_path, features_path, train=True)
    validation_anet = ANetCaptionsDataset(anet_path, features_path, train=False)
    # print("Training size: {}, Validation Size: {}".format(len(train_anet), len(validation_anet)))

    train_anet_generator = data.DataLoader(train_anet, batch_size=64, num_workers=8)
    validation_anet_generator = data.DataLoader(validation_anet)

    num_epochs = 25
    learning_rate = 1e-5
    visual_feature_size = train_anet.get_max_fm_size() * 1024
    lstm_hidden_size = 50
    vocab_size = train_anet.get_vocab_size()
    net = DecoderLSTM(visual_feature_size, lstm_hidden_size, vocab_size).to(device)
    opt = optim.SGD(params=net.parameters(), lr=learning_rate)
    loss = nn.NLLLoss()

    print("Visual Feature Size: {}".format(visual_feature_size))
    print("Vocab Size: {}".format(vocab_size))

    for epoch in range(num_epochs):
        print("Started Epoch {}".format(epoch))
        epoch_summary = run_single_epoch(data_loader=train_anet_generator, model=net, optimizer=opt, criterion=loss, batch_size=32)
        print("Epoch Loss: {}".format(epoch_summary))
