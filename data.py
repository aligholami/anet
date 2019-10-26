import json
from torch.utils.data.dataset import Dataset


class ANetCaptionsDataset(Dataset):
    def __init__(self, anet_json_path, train=True):
        self.anet_contents = self.read_json_file(anet_json_path)
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
            else:
                subset_dict = {key: value for key, value in content_dict['database'].items() if
                               value['subset'] == 'validation'}

            return subset_dict

        if train == True:
            self.anet_subset = get_subset(self.anet_contents, 'training')
        else:
            self.anet_subset = get_subset(self.anet_contents, 'validation')

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass

    def read_json_file(self, path_to_file):
        """
        Read the json file and return it as a Python dictionary.
        :param path_to_file: string path to the JSON file.
        :return: a Python dictionary
        """
        with open(path_to_file, 'r') as fo:
            return json.load(fo)


anet_path = '../gvd-data/ActivityNet/data/anet/anet_annotations_trainval.json'
train_anet_captions = ANetCaptionsDataset(anet_path, train=True)
validation_anet_captions = ANetCaptionsDataset(anet_path, train=False)
print(len(train_anet_captions.anet_subset))
print(len(validation_anet_captions.anet_subset))
