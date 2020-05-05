#!/usr/bin/python3

import torch
import torch.nn as nn
import torch.nn.functional as F

''' Convolution/Decovolution architecture for encoder/decoder '''
class ConvDeconv(object):
    class Down(nn.Conv1d):
        def __init__(self, channels_in, channels_out, kernel_size = 1, stride = 1, padding = 0):
            super(ConvDeconv.Down, self).__init__(channels_in, channels_out, kernel_size = kernel_size, stride = stride, padding = padding)

    class Up(nn.ConvTranspose1d):
        def __init__(self, channels_in, channels_out, kernel_size = 1, stride = 1, padding = 0):
            super(ConvDeconv.Up, self).__init__(channels_in, channels_out, kernel_size = kernel_size, stride = stride, padding = padding, output_padding = stride - 1 if stride > 1 else 0)

''' max-pooling/upsampling architecture for encoder/decoder '''
class PoolConv(object):
    class Down(nn.Conv1d):
        def __init__(self, channels_in, channels_out, kernel_size = 1, stride = 1, padding = 0):
            super(PoolConv.Down, self).__init__(channels_in, channels_out, kernel_size = kernel_size, stride = 1, padding = padding)
            self.pool = stride
        def forward(self, x):
            x = super(PoolConv.Down, self).forward(x)
            x = F.max_pool1d(x, kernel_size = self.pool)
            return x

    class Up(nn.Conv1d):
        def __init__(self, channels_in, channels_out, kernel_size = 1, stride = 1, padding = 0):
            super(PoolConv.Up, self).__init__(channels_in, channels_out, kernel_size = kernel_size, stride = 1, padding = padding)
            self.upsample = stride
        def forward(self, x):
            x = F.interpolate(x, scale_factor = self.upsample)
            x = super(PoolConv.Up, self).forward(x)
            return x


''' network base class '''
class Net(nn.Module):

    def __init__(self, name):
        super(Net, self).__init__()
        self.name = name

    def save(self, model_dir, suffix = None):
        self.cpu()
        model_file = model_dir + '/' + self.name + ('' if suffix is None else '.' + suffix) + '.net'
        torch.save(self.state_dict(), model_file)
        self.cuda()

    def load(self, model_dir, suffix = None):
        self.cpu()
        model_file = model_dir + '/' + self.name + ('' if suffix is None else '.' + suffix) + '.net'
        self.load_state_dict(torch.load(model_file))
        self.cuda()


class TcnBlock(Net):

    def __init__(self, name = 'tcn_block', channels = 64, conv_len = 3, stride = 2, conv_type = PoolConv.Down):
        super(TcnBlock, self).__init__(name)
        self.stride = stride
        self.conv_len = conv_len
        self.conv1 = conv_type(channels, channels, kernel_size = conv_len, stride = self.stride, padding = conv_len // 2)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size = 1)
        self.cuda()
        
    def forward(self, x):
        y = F.relu(self.conv1(x))

        if y.shape[-1] < x.shape[-1]:
            y = F.max_pool1d(x, kernel_size = self.stride, padding=(0 if x.shape[-1] % 2 == 0 else 1)) + y
        elif y.shape[-1] > x.shape[-1]:
            y = F.interpolate(x, scale_factor = self.stride) + y
        else:
            y = y + x
        return y


class Tcn(Net):

    def __init__(self, input_dim, output_dim, channels = 64, conv_len = 3, stride = [], layers = 10, name = 'tcn', conv_type = PoolConv, add_avg_pool=False):
        super(Tcn, self).__init__(name)
        self.add_avg_pool = add_avg_pool
        self.stride = stride
        self.layers = layers
        self.conv_len = conv_len
        self.conv_in = nn.Conv1d(input_dim, channels, kernel_size = 1)
        self.conv_out = nn.Conv1d(channels, output_dim, kernel_size = 1)
        self.output_dim = output_dim
        self.encoder_blocks = []
        self.decoder_blocks = []
        assert len(stride) == layers
        for l in range(layers):
            self.encoder_blocks += [ TcnBlock(channels = channels, conv_len = conv_len, stride = stride[l], conv_type = conv_type.Down) ]
        self.encoder_blocks = nn.ModuleList(self.encoder_blocks)
        self.cuda()

    def forward(self, x, is_eval=False):
        if self.add_avg_pool:
            x = F.avg_pool1d(x, kernel_size=10, stride=1, padding=5)
            x = x[:, :, :-1]
        length = x.shape[-1]
        x = self.conv_in(x)
        for enc in self.encoder_blocks:
            x = enc(x)
        latent = x
        x = self.conv_out(x)
        x = F.log_softmax(x, dim = 1)
        if is_eval:
            x = F.interpolate(x, size=(length))
        return x, latent


class Dummy(Net):

    def __init__(self, input_dim, name = 'dummy'):
        super(Dummy, self).__init__(name)
        self.dummy = nn.Conv1d(input_dim, input_dim, 1)
        self.cuda()

    def forward(self, x):
        x = self.dummy(x)
        x = torch.log_softmax(x, dim = 1)
        return x, x

