import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils import data
from data import ANetCaptionsDataset
from model import DecoderLSTM
from tqdm import tqdm
from contextlib import ExitStack

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")


def run_single_epoch(data_loader, model, optimizer, criterion, prefix='train'):
    """
    Run the model for a single epoch.
    :param prefix: An string specifying the state of the function. Either training ('train') or validation ('val').
    :param data_loader: Customized PyTorch inherited data loader for the dataset.
    :param model: Model to train.
    :param optimizer: Optimization technique.
    :param criterion: Loss function to use.
    :return: Epoch summary.
    """

    cm = None
    if prefix == 'train':
        model.train()
        cm = ExitStack
    elif prefix == 'val':
        model.eval()
        cm = torch.no_grad

    else:
        print("Invalid prefix, aborting the process.")

    with cm():
        total_loss = 0.0
        iteration = 0
        for x, target_description in tqdm(data_loader):
            iter_loss = 0
            vf = x[2]
            decoder_input = vf.view(-1, vf.shape[1] * vf.shape[2])
            decoder_h = model.init_hidden().to(device)
            decoder_c = model.init_hidden().to(device)
            x_type = 'vis'
            optimizer.zero_grad()

            # Teacher forced decoder training
            for idx in range(len(target_description)):
                predictions, (decoder_h, decoder_c) = model(decoder_input.to(device), decoder_h,
                                                            decoder_c, x_type)
                iter_loss += criterion(predictions, target_description[idx].to(device))
                decoder_input = target_description[idx]
                x_type = 'lan'

            iteration += 1

            if prefix == 'train':
                iter_loss.backward()
                optimizer.step()

            total_loss += iter_loss

    return total_loss / len(data_loader)


if __name__ == '__main__':
    anet_path = '../gvd-data/ActivityNet/data/anet/anet_annotations_trainval.json'
    features_path = '../gvd-data/ActivityNet/data'
    train_anet = ANetCaptionsDataset(anet_path, features_path, train=True)
    validation_anet = ANetCaptionsDataset(anet_path, features_path, train=False)
    # print("Training size: {}, Validation Size: {}".format(len(train_anet), len(validation_anet)))

    train_anet_generator = data.DataLoader(train_anet, batch_size=128, num_workers=6)
    validation_anet_generator = data.DataLoader(validation_anet, batch_size=128, num_workers=6)

    num_epochs = 25
    learning_rate = 1e-5
    visual_feature_size = train_anet.max_vid_fm_size[0] * train_anet.max_vid_fm_size[1]
    lstm_hidden_size = 256
    vocab_size = train_anet.vocab_size
    net = DecoderLSTM(visual_feature_size, lstm_hidden_size, vocab_size).to(device)
    opt = optim.SGD(params=net.parameters(), lr=learning_rate)
    loss = nn.NLLLoss()

    print("Visual Feature Size: {}*{}={}".format(train_anet.max_vid_fm_size[0], train_anet.max_vid_fm_size[1], visual_feature_size))
    print("Vocab Size: {}".format(vocab_size))

    for epoch in range(num_epochs):
        print("\n\nStarted Epoch {}".format(epoch))
        epoch_summary = run_single_epoch(data_loader=train_anet_generator, model=net, optimizer=opt, criterion=loss,
                                         prefix='train')
        print("Training Loss: {}".format(epoch_summary))
        epoch_summary = run_single_epoch(data_loader=validation_anet_generator, model=net, optimizer=opt,
                                         criterion=loss, prefix='val')
        print("Validation Loss: {}".format(epoch_summary))
