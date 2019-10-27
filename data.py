import numpy as np
import json
import os
from torch.utils.data.dataset import Dataset


class ANetCaptionsDataset(Dataset):
    def __init__(self, anet_json_path, features_root, train=True):
        self.anet_contents = self.read_json_file(anet_json_path)
        self.anet_subset = {}
        self.features_root = features_root

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
                self.features_root = os.path.join(self.features_root, 'training')

            else:
                subset_dict = {key: value for key, value in content_dict['database'].items() if
                               value['subset'] == 'validation'}
                self.features_root = os.path.join(self.features_root, 'validation')

            return subset_dict

        def create_x_label_pairs(data_dict):
            """
            Create a list of tuples. Each tuple corresponds to a sample. The format for each tuple is (X, LABEL).
            :param data_dict: A train/validation subset of data. It is a dictionary with video ids as keys.
            :return: A list of (X, LABEL) tuples.
            """
            x_label_pairs = []
            init_vid_features = np.array([])
            for vid_key, vid_val in data_dict.items():
                vid_annotations = vid_val['annotations']
                vid_duration = vid_val['duration']
                for annotation in vid_annotations:
                    x_label_pair = ()
                    segment_start = annotation['segment'][0]
                    segment_end = annotation['segment'][1]
                    description = annotation['sentence']
                    x = (vid_key, vid_duration, init_vid_features, segment_start, segment_end)
                    label = description

                    x_label_pairs.append((x, label))

            return x_label_pairs

        if train == True:
            self.anet_subset = get_subset(self.anet_contents, 'training')
        else:
            self.anet_subset = get_subset(self.anet_contents, 'validation')

        self.anet_subset = create_x_label_pairs(self.anet_subset)

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

        feature_path = os.path.join(self.features_root, vid_key + '_bn.npy')
        vid_features = np.load(feature_path)
        new_x = (x[0], x[1], vid_features, x[2], x[3])

        return new_x, label

    @staticmethod
    def read_json_file(path_to_file):
        """
        Read the json file and return it as a Python dictionary.
        :param path_to_file: string path to the JSON file.
        :return: a Python dictionary
        """
        with open(path_to_file, 'r') as fo:
            return json.load(fo)


anet_path = '../gvd-data/ActivityNet/data/anet/anet_annotations_trainval.json'
features_path = '../gvd-data/ActivityNet/data'
train_anet_captions = ANetCaptionsDataset(anet_path, features_path, train=True)
validation_anet_captions = ANetCaptionsDataset(anet_path, features_path, train=False)

print("Training size: {}, Validation Size: {}".format(len(train_anet_captions), len(validation_anet_captions)))
for x, label in train_anet_captions:
    print("Vid Key: {}".format(x[0]))
    print("Vid Duration: {}".format(x[1]))
    print("Feature Map Size: {}".format(x[2].shape))
    print("(Start, End): ({}, {})".format(x[3], x[4]))
    print("Description: {}".format(label))