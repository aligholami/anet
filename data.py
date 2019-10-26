import json


def read_json_file(path_to_file):
    """
    Read the json file and return it as a Python dictionary.
    :param path_to_file: string path to the JSON file.
    :return: a Python dictionary
    """
    with open(path_to_file, 'r') as fo:
        return json.load(fo)



anet_json_path = '../gvd-data/ActivityNet/data/anet/anet_annotations_trainval.json'
anet_contents = read_json_file(anet_json_path)
print(type(anet_contents))