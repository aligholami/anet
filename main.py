from data import ANetCaptionsDataset

if __name__ == '__main__':
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