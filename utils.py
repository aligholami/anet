import json

class SubmissionHandler:
    def __init__(self, ix2w):
        """
        This class initializes a copy of index to word dictionary to handle process the predicted sentences containing word ids.
        :param ix2w: a dictionary, mapping indexes to words. Note that the ix2w is global among train and validation.
        """
        self.ix2w = ix2w

    def get_words_from_indexes(self, indexes):
        """
        converts a bunch of word token indexes to actual word strings.
        :param indexes: a list of indexes.
        :return: a list of words.
        """

        word_list = [self.ix2w[index] for index in indexes]

        return word_list

    def create_submission_file(self, results, path):
        """
        Create a submission file (JSON) for evaluation purposes, given a dictionary of predictions and ix to word dictionary.
        :param results: a dictionary with vid_key as keys. For each vid_key there are two elements: "sentence": string and
        "time_segment": [start(float), end(float)].
        :return: Nothing.
        """

        submission_dict = {
            "version": "VERSION 1.0",
            "results": results,
            "external_data": {}
        }

        with open(path, 'w') as fd:
            json.dump(submission_dict, fd)

        print(f"Generated submission file: {path}")



