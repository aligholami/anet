import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils import data
from data import ANetCaptionsDataset
from data import ANetCaptionsConstants
from model import DecoderLSTM
from tqdm import tqdm
from contextlib import ExitStack
from utils import SubmissionHandler

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")


def results_list_to_dict(results_list, submission_handler):
    """
    Takes a list of mini batches and returns a single dictionary with video keys as keys and a list as values.
    Each of these lists contains multiple dictionaries with "sentence" and "seg start-end" keys.
    :param submission_handler: The submission handler object. Gives access to the ix2w dictionary.
    :param results_list: a list of mini batch results.
    :return: A dictionary of submission format.
    (similar to https://github.com/aligholami/densevid_eval_spice/blob/master/sample_submission.json)
    NOTE: The sentences are not converted to words and are still ids.
    """
    results_dict = {}

    for mini_batch_result in results_list:
        for ix, vid_key in enumerate(mini_batch_result['vid_keys']):
            try:
                key_arr = results_dict[vid_key]
            except KeyError as ke:
                key_arr = []

            sentence = submission_handler.get_words_from_indexes(mini_batch_result["sentence_ids"].tolist()[ix])
            print(f"Predicted Description: {sentence}")
            key_arr.append({
                "sentence": sentence,
                "timestamp": [mini_batch_result["seg_starts"].tolist()[ix], mini_batch_result["seg_ends"].tolist()[ix]]
            })

            results_dict[vid_key] = key_arr

    return results_dict


def run_single_epoch(data_loader, model, optimizer, criterion, submission_handler, prefix='train'):
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

    summary = {
        "loss": 0,
        "results": {}
    }

    with cm():
        total_loss = 0.0
        iteration = 0
        results = []
        for x, target_description in tqdm(data_loader):
            iter_loss = 0
            vf = x[2]
            batch_size = vf.size(0)
            SOS_TENSOR = torch.empty((batch_size, 1))
            SOS_TENSOR[:] = torch.tensor([ANetCaptionsConstants.SOS_TOKEN_IDX])
            decoder_input = SOS_TENSOR
            decoder_h = model.init_hidden(batch_size, vf.to(device)).to(device)
            decoder_c = model.init_cell(batch_size).to(device)
            optimizer.zero_grad()

            sentence_ids = []
            # Teacher forced decoder training
            for idx in range(len(target_description)):
                predictions, (decoder_h, decoder_c) = model(decoder_input.to(device), decoder_h, decoder_c)
                iter_loss += criterion(predictions, target_description[idx].to(device))
                decoder_input = target_description[idx]

                # take the best word ids
                word_ids = predictions.argmax(dim=1)
                sentence_ids.append(word_ids.unsqueeze(dim=1))  # Unsqueeze()?: Concat on dim=1 later

            mini_batch_results = {
                "vid_keys": x[0],
                "sentence_ids": torch.cat(sentence_ids, dim=1),
                "seg_starts": x[3],
                "seg_ends": x[4]
            }

            results.append(mini_batch_results)

            iteration += 1
            if prefix == 'train':
                iter_loss.backward()
                optimizer.step()

            total_loss += iter_loss

    # have a separate fcn to convert mini batch results to dict -> better performance (higher gpu util)
    results = results_list_to_dict(results, submission_handler)

    summary['loss'] = total_loss / len(data_loader)
    summary['results'] = results

    return summary


if __name__ == '__main__':
    anet_path = '/local-scratch/ActivityNet/annotations/reformatted-annotations/anet/anet_annotations_trainval.json'
    features_path = '/local-scratch/ActivityNet/vid-frame-wise-features/rgb_motion_1d'
    train_anet = ANetCaptionsDataset(anet_path, features_path, train=True)
    validation_anet = ANetCaptionsDataset(anet_path, features_path, train=False)
    submission_handler = SubmissionHandler(train_anet.idx2word)

    # print("Training size: {}, Validation Size: {}".format(len(train_anet), len(validation_anet)))

    train_anet_generator = data.DataLoader(train_anet, batch_size=128, num_workers=6)
    validation_anet_generator = data.DataLoader(validation_anet, batch_size=128, num_workers=6)

    num_epochs = 25
    learning_rate = 1e-5
    visual_feature_size = train_anet.max_vid_fm_size[0] * train_anet.max_vid_fm_size[1]
    lstm_hidden_size = 256
    vocab_size = train_anet.vocab_size
    net = DecoderLSTM(visual_feature_size, lstm_hidden_size, vocab_size, device).to(device)
    opt = optim.SGD(params=net.parameters(), lr=learning_rate)
    loss = nn.NLLLoss()

    print("Visual Feature Size: {}*{}={}".format(train_anet.max_vid_fm_size[0], train_anet.max_vid_fm_size[1],
                                                 visual_feature_size))
    print("Vocab Size: {}".format(vocab_size))

    for epoch in range(num_epochs):
        print("\n\nStarted Epoch {}".format(epoch))
        epoch_summary = run_single_epoch(data_loader=train_anet_generator, model=net, optimizer=opt, criterion=loss,
                                         submission_handler=submission_handler, prefix='train')
        print("Training Loss: {}".format(epoch_summary['loss']))
        epoch_summary = run_single_epoch(data_loader=validation_anet_generator, model=net, optimizer=opt,
                                         criterion=loss, submission_handler=submission_handler, prefix='val')
        print("Validation Loss: {}".format(epoch_summary['loss']))
