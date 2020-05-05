#!/usr/bin/python3

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

from utils.datareader import DataReader
from utils.net import Tcn, PoolConv, ConvDeconv, Dummy
from utils.trainer import Trainer, Forwarder
from utils.eval import Scorer, MapScorer, thumos_output
from utils.qualitative_res import save_qualitative_result

iteration = 10000
only_labels = False

with open('data/testset', 'r') as f:
    video_list = f.read().split('\n')[0:-1]

dataset = DataReader([video_list[0]], only_labels = only_labels, feature_dir = 'i3d_bow')
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


forwarder = Forwarder(net)

i = iteration
net.cpu()
net.load('results', suffix = 'iter-' + str(i))
#net.load('results', suffix = 'check')

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
colors_dict = {0:'grey', 1:'blue', 2:'red', 3:'green', 4:'yellow', 5: 'brown', 6:'orange',
               7:'deeppink', 8:'magenta', 9:'limegreen', 10:'cyan', 11: 'dodgerblue', 12: 'purple',
               13: 'deepskyblue', 14: 'chocolate', 15: 'yellowgreen', 16: 'indianred', 17: 'khaki', 
               18: 'darkcyan', 19: 'aquamarine', 20: 'lightgreen'}
for key, value in colors_dict.items():
    colors_dict[key] = colors[value]

net.cuda()
for video in video_list:
    if video == 'video_test_0000793':
        continue
    dataset = DataReader([video], only_labels = only_labels, feature_dir = 'i3d_bow')
    data, labels = dataset[0]
    scores = forwarder.forward(data)
    #save_qualitative_result(labels.tolist(), np.argmax(scores, axis=1).tolist(), video, 'qualitative_results', colors_dict)
    thumos_output('results/' + video, scores)

