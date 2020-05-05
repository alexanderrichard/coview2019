#!/usr/bin/python3.5

import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class LabelSampler(object):
    def __init__(self, strides):
        self.factor = 1
        for stride in strides:
            self.factor *= stride

    def sample(self, labels):
        result = []
        labels = labels.squeeze()
        for i in range(0, len(labels), self.factor):
            temp_labels = labels[i:(i + self.factor if (i + self.factor) < len(labels) else len(labels))]
            result.append(np.argmax(np.bincount(temp_labels)))
        labels = labels.unsqueeze(0)
        return torch.tensor(result).view(1, -1)

class Trainer(object):

    def __init__(self, dataset, net, learning_rate = 0.01):
        self.dataset = dataset
        self.net = net
        self.prev_loss = 0
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = learning_rate)
        self.dataloader = DataLoader(dataset, batch_size = 1, shuffle = True, num_workers = 1)
        self.net.train()

    def loss(self, output, labels, latent):
        output = output.transpose(1,2)
        output = output.view(-1, output.shape[2])
        labels = labels.view(-1)
        ce = F.nll_loss(output, labels)
        stats = {'ce': float(ce.item())}
        return ce, stats

    def train_iteration(self, data, labels):
        data = data.cuda()
        labels = labels.cuda()
        self.optimizer.zero_grad()
        # forward
        output, latent = self.net(data)
        # backprop
        loss, stats = self.loss(output, labels, latent)
        loss.backward()
        self.optimizer.step()
        return stats

    def train(self, result_dir, iterations, save_frequency, strides=[]):
        iter_idx = 0
        ls = LabelSampler(strides)
        while iter_idx < iterations:
            for i, (data, labels) in enumerate(self.dataloader):
                data = data.transpose(1,2)
                labels = ls.sample(labels)
                stats = self.train_iteration(data, labels)
                iter_idx += 1
                # some output, model saving, etc
                if iter_idx % 10 == 0:
                    print('iter-' + str(iter_idx) + '    ' + '    '.join([ '%s: %.4f' % (key, stats[key]) for key in stats]))
                if save_frequency > 0 and iter_idx % save_frequency == 0:
                    self.net.save(result_dir, suffix = 'iter-' + str(iter_idx))
                if iter_idx == iterations:
                    break
        self.net.save(result_dir)


class Forwarder(object):

    def __init__(self, net):
        self.net = net
        self.net.eval()

    def forward(self, data):
        data = torch.FloatTensor(data).unsqueeze(0).cuda()
        data = data.transpose(1,2)
        data = data.cuda()
        with torch.no_grad():
            output, latent = self.net(data, is_eval=True)
        scores = output.transpose(1,2).squeeze(0).cpu().numpy()
        return scores


