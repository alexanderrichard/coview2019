#!/usr/bin/python3

import numpy as np
from utils.datareader import DataReader
from utils.net import Tcn, PoolConv, ConvDeconv, Dummy
from utils.trainer import Trainer, Forwarder
from utils.eval import Scorer, MapScorer, thumos_output

iterations = 50000
save_frequency = 2000
only_labels = False
map_threshold = 0.3

### training ###
with open('data/trainset', 'r') as f:
    video_list = f.read().split('\n')[0:-1]
dataset = DataReader(video_list, only_labels = only_labels, feature_dir = 'i3d_bow')

num_layers = 20
strides = [1] * num_layers

net = Tcn(input_dim = dataset.feature_dim(),
          output_dim = dataset.num_classes(),
          channels = 128,
          conv_len = 3,
          stride = strides,
          layers = num_layers,
          name = 'tcn',
          conv_type = PoolConv)


trainer = Trainer(dataset, net, learning_rate = 0.001)
trainer.train(result_dir = 'results',
              iterations = iterations,
              save_frequency = save_frequency, strides = strides)


### evaluation ###
for part in ['trainset']:

    print('\n\n### %s ###' % part)
    with open('data/' + part, 'r') as f:
        video_list = f.read().split('\n')[0:-1]

    forwarder = Forwarder(net)

    for i in range(save_frequency, iterations+1, save_frequency):
        scorer = Scorer()
        map_scorer = MapScorer()
        net.cpu()
        net.load('results', suffix = 'iter-' + str(i))
        net.cuda()
        for video in video_list:
            if video == 'video_test_0000793':
                continue
            dataset = DataReader([video], only_labels = only_labels, feature_dir = 'i3d_bow')
            data, labels = dataset[0]
            scores = forwarder.forward(data)
            scorer.add_sequence(scores, labels)
            map_scorer.add_sequence(scores, labels)
        print('iter-%d:    accuracy: %.4f    ce_loss: %.4f    mAP@%.1f: %.4f' % (i, scorer.frame_accuracy(), scorer.cross_entropy(), map_threshold, map_scorer.mAP(map_threshold)))

