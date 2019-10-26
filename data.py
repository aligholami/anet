import json
from torch.utils.data.dataset import Dataset


class ANetCaptionsDataset(Dataset):
    def __init__(self, anet_json_path):
        self.anet_contents = self.read_json_file(anet_json_path)

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


anet_json_path = '../gvd-data/ActivityNet/data/anet/anet_annotations_trainval.json'
anet_captions = ANetCaptionsDataset(anet_json_path)