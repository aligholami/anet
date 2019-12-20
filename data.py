import numpy as np
import torch
import json
import os
from torch.utils.data.dataset import Dataset
from tqdm import tqdm


class ANetCaptionsConstants:
    SOS_TOKEN_IDX = 0
    EOS_TOKEN_IDX = 1


class ANetCaptionsDataset(Dataset):
    def __init__(self, anet_json_path, features_root, train=True):
        self.anet_contents = self.read_json_file(anet_json_path)

        # global word2idx, idx2word
        self.word2idx, self.idx2word = self.get_word2idx_idx2word(self.anet_contents)
        self.one_hot_size = self.get_vocab_size()
        self.features_root = features_root
        if train:
            self.features_subset = 'training'
        else:
            self.features_subset = 'validation'
        self.fm_post_fix = '_bn.npy'  # possible options for now: _bn | _resnet
        self.anet_subset = {}

        def get_subset(content_dict, subset):
            """
            Return a subset of the dataset (train or validation) based on the subset type.
            :param content_dict: Raw dataset dictionary.
            :param subset: String either 'training' or 'validation'.
            :return: A dictionary containing the desired subset.
            """
            subset_dict = {}
            if subset == 'training':
                subset_dict = {key: value for key, value in content_dict['database'].items() if
                               value['subset'] == 'training'}
            elif subset == 'validation':
                subset_dict = {key: value for key, value in content_dict['database'].items() if
                               value['subset'] == 'validation'}
            else:
                print("Invalid data subset selected.")
                exit(0)

            return subset_dict

        def create_x_label_pairs(anet_subset):
            """
            Create a list of tuples. Each tuple corresponds to a sample. The format for each tuple is (X, LABEL).
            :param anet_subset: A train/validation subset of data. It is a dictionary with video ids as keys.
            :return: A list of (X, LABEL) tuples.
            """
            x_label_pairs = []
            init_vid_features = np.array([])
            for vid_key, vid_val in anet_subset.items():
                vid_annotations = vid_val['annotations']
                vid_duration = vid_val['duration']

                for annotation in vid_annotations:
                    segment_start = annotation['segment'][0]
                    segment_end = annotation['segment'][1]
                    description = annotation['sentence']
                    x = (vid_key, vid_duration, init_vid_features, segment_start, segment_end)
                    label = [self.word2idx[word] for word in description.strip().split()]
                    label = label + [self.word2idx['<EOS>']]

                    x_label_pairs.append((x, label))

            return x_label_pairs

        if train:
            self.anet_subset = get_subset(self.anet_contents, 'training')
        else:
            self.anet_subset = get_subset(self.anet_contents, 'validation')

        self.anet_subset = create_x_label_pairs(self.anet_subset)
        self.max_vid_fm_size = self.get_max_fm_size()
        self.vocab_size = self.get_vocab_size()

    def __len__(self):
        return len(self.anet_subset)

    def __getitem__(self, idx):
        """
        To be more ram efficient, we need to load the features while we need them.
        :param idx: target sample index.
        :return: (X, LABEL) tuple with updated feature.
        """
        x, label = self.anet_subset[idx]
        vid_key = x[0]

        feature_path = os.path.join(self.features_root, vid_key + self.fm_post_fix)

        try:
            vid_features = np.load(feature_path)
        except BaseException as bex:
            # print("Not Found, Skipping...")
            vid_features = np.random.rand(100, 1024)

        # print(f"Maximum feature map size: {self.max_vid_fm_size}")
        # print(f"Vid feature map size: {vid_features.shape}")

        padded_vid_features = self.zero_pad_feature_map(vid_features, self.max_vid_fm_size)
        new_x = (x[0], x[1], padded_vid_features, x[3], x[4])

        return new_x, label

    def get_vocab_size(self):
        """
        Vocab size getter.
        :return: Vocab size, an integer.
        """
        return len(self.word2idx.keys())

    def get_max_fm_size(self):
        """
        Feature map size getter.
        :return: Maximum feature map size along the whole dataset (train and validation) an integer.
        """

        def get_fm_size(key_subset, vid_key):
            """
            Find the feature map size for a specific video key.
            :param key_subset: Training/Validation Subset.
            :param vid_key: An integer, desired key.
            :return: Video feature map size.
            """
            feature_path = os.path.join(self.features_root, vid_key + self.fm_post_fix)

            try:
                fm_shape = np.load(feature_path).shape
            except BaseException as bex:
                fm_shape = (1, 1024)

            return fm_shape

        max_vid_shape_temporal = 0  # Variable based on the temporal dimension of the each sample video
        vid_shape_spatial = 0  # Constant for all data based on the dataset
        print("Looking for the maximum feature map size on {} subset.".format(self.features_subset))
        for vid_key, vid_val in tqdm(self.anet_contents['database'].items()):
            vid_shape_temporal, vid_shape_spatial = get_fm_size(vid_val['subset'], vid_key)

            if vid_shape_temporal > max_vid_shape_temporal:
                max_vid_shape_temporal = vid_shape_temporal

        return max_vid_shape_temporal, vid_shape_spatial

    @staticmethod
    def get_word2idx_idx2word(anet_contents):
        """
        Designed to grab unique words and their index (along both train and validation).
        :param anet_contents: ANetCaption dictionary.
        :return: dictionary of format {word: idx}, {idx: word}
        """

        def add_sos_eos_tokens(word_list):
            return ['<SOS>'] + ['<EOS>'] + word_list

        all_words = []
        anet_contents = anet_contents['database']
        for vid_key, vid_val in anet_contents.items():
            vid_annotations = vid_val['annotations']
            for annotation in vid_annotations:
                description = annotation['sentence']
                description_words = description.strip().split(' ')
                all_words += description_words
        unique_words = add_sos_eos_tokens(list(set(all_words)))

        word2idx = {word: key for key, word in enumerate(unique_words)}
        idx2word = {key: word for key, word in enumerate(unique_words)}

        return word2idx, idx2word

    @staticmethod
    def read_json_file(path_to_file):
        """
        Read the json file and return it as a Python dictionary.
        :param path_to_file: string path to the JSON file.
        :return: a Python dictionary
        """
        with open(path_to_file, 'r') as fo:
            return json.load(fo)

    @staticmethod
    def zero_pad_feature_map(fm, target_shape):
        """
        Pads a feature map fm with zeros to reach the target shape.
        :param fm: feature map to be padded.
        :param target_shape: target shape.
        :return: padded feature map.
        """
        padded_fm = np.zeros((target_shape[0], 1024))
        padded_fm[:fm.shape[0], :] = fm

        return padded_fm
