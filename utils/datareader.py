#!/usr/bin/python3

import numpy as np

class DataReader(object):

    def __init__(self, video_list, only_labels = False, feature_dir = 'features'):
        self.idx = 0
        self.only_labels = only_labels
        # read mapping from label to index
        label2index = dict()
        with open('data/mapping.txt', 'r') as f:
            lines = f.read().split('\n')[0:-1]
            for line in lines:
                label2index[line.split()[1]] = int(line.split()[0])
        self.n_classes = len(label2index)
        # read labels
        self.labels = []
        for video in video_list:
            with open('data/groundTruth/' + video + '.txt', 'r') as f:
                labels = f.read().split('\n')[0:-1]
                self.labels.append( np.array([label2index[l] for l in labels], dtype = np.int64) )
        # read features if required
        if not only_labels:
            self.features = []
            for video in video_list:
                with open('data/groundTruth/' + video + '.txt', 'r') as f:
                    features = np.transpose(np.load('data/' + feature_dir + '/' + video + '.npy'))
                    self.features.append(features)

    def feature_dim(self):
        if self.only_labels:
            return self.n_classes
        else:
            return self.features[0].shape[1]

    def num_classes(self):
        return self.n_classes

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.only_labels:
            one_hot = np.zeros((self.labels[idx].shape[0], self.n_classes), dtype = np.float32)
            one_hot[np.arange(one_hot.shape[0]), self.labels[idx]] = 1
            return one_hot, self.labels[idx]
        else:
            return self.features[idx], self.labels[idx]

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx == len(self):
            self.idx = 0
            raise StopIteration
        else:
            self.idx += 1
            return self[self.idx - 1]

