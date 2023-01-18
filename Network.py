# YOUR CODE HERE
# import tensorflow as tf
# import torch

"""This script defines the network.
"""
import torch
import torch.nn as nn
import torch.nn.functional as nnFunctional
import math

# import torch.optim as optim
# from torch.autograd import Variable
# import torchvision.datasets as dset
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# import torchvision.models as models
# import sys

# class MyNetwork(object):

#     def __init__(self, configs):
#         self.configs = configs

#     def __call__(self, inputs, training):
#         '''
#         Args:
#             inputs: A Tensor representing a batch of input images.
#             training: A boolean. Used by operations that work differently
#                 in training and testing phases such as batch normalization.
#         Return:
#             The output Tensor of the network.
#         '''
#         return self.build_network(inputs, training)

#     def build_network(self, inputs, training):
#         return inputs

class Bottleneck_Block(nn.Module):
    def __init__(self, nChannels, growthRate, dropRate):
        super(Bottleneck_Block, self).__init__()
        interChannels = 4*growthRate
        self.batchNormalization_1 = nn.BatchNorm2d(nChannels)
        self.convolution1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.batchNormalization_2 = nn.BatchNorm2d(interChannels)
        self.convolution2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)
        self.droprate = dropRate                       

    def forward(self, x):
        out = self.convolution1(nnFunctional.relu(self.batchNormalization_1(x)))
        if self.droprate > 0:
            out = nnFunctional.dropout(out, p=self.droprate, training=self.training)
        out = self.convolution2(nnFunctional.relu(self.batchNormalization_2(out)))
        if self.droprate > 0:
            out = nnFunctional.dropout(out, p=self.droprate, training=self.training)
        out = torch.cat((x, out), 1)
        return out

class SingleLayer_Block(nn.Module):
    def __init__(self, nChannels, growthRate, dropRate):
        super(SingleLayer_Block, self).__init__()
        self.batchNormalization_1 = nn.BatchNorm2d(nChannels)
        self.convolution1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)
        self.droprate = dropRate                       

    def forward(self, x):
        out = self.convolution1(nnFunctional.relu(self.batchNormalization_1(x)))
        if self.droprate > 0:
            out = nnFunctional.dropout(out, p=self.droprate, training=self.training)
        out = torch.cat((x, out), 1)
        return out

class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels, dropRate):
        super(Transition, self).__init__()
        self.batchNormalization_1 = nn.BatchNorm2d(nChannels)
        self.convolution1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)
        self.droprate = dropRate                       

    def forward(self, x):
        out = self.convolution1(nnFunctional.relu(self.batchNormalization_1(x)))
        if self.droprate > 0:
            out = nnFunctional.dropout(out, p=self.droprate, training=self.training)
        out = nnFunctional.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    # growthRate=12, depth=100, compression_factor=0.5, bottleneck=True, nClasses=10
    # def __init__(self, growthRate, depth, compression_factor, nClasses, bottleneck, dropRate=0.2):
    def __init__(self, growthRate, depth, compression_factor, nClasses, bottleneck, dropRate=0):
        super(DenseNet, self).__init__()

        nDenseBlocks = (depth-4) // 3 # (100 - 4) // 3 = 32
        if bottleneck:
            # 32 // 2 = 16
            nDenseBlocks //= 2

        nChannels = 2*growthRate # nChannels = 2 * 12 = 24
        self.convolution1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1,
                               bias=False)
        # self.denseblock1 = self.stack_dense_layer(24,        12,          16,          True)
        self.denseblock1 = self.stack_dense_layer(nChannels, growthRate, nDenseBlocks, bottleneck, dropRate)
        nChannels += nDenseBlocks*growthRate # nChannels = 24 + (16 * 12) = 216
        nOutChannels = int(math.floor(nChannels*compression_factor)) # nOutChannels = 216 * 0.5 = 108
        # self.transitionblock1 = Transition(216, 108)
        self.transitionblock1 = Transition(nChannels, nOutChannels, dropRate)

        nChannels = nOutChannels # nChannels = 108
        # self.denseblock2 = self.stack_dense_layer(108,         12,         16,          True)
        self.denseblock2 = self.stack_dense_layer(nChannels, growthRate, nDenseBlocks, bottleneck, dropRate)
        nChannels += nDenseBlocks*growthRate # nChannels = 108 + (16 * 12) = 300
        nOutChannels = int(math.floor(nChannels*compression_factor)) # nOutChannels = 300 / 2 = 150
        # self.transitionblock2 = Transition(300, 150)
        self.transitionblock2 = Transition(nChannels, nOutChannels, dropRate)

        nChannels = nOutChannels # nChannels = 150
        # self.denseblock3 = self.stack_dense_layer(150,         12,          16,         True)
        self.denseblock3 = self.stack_dense_layer(nChannels, growthRate, nDenseBlocks, bottleneck, dropRate)
        nChannels += nDenseBlocks*growthRate # nChannels = 150 + (16 * 12) = 342

        self.batchNormalization_1 = nn.BatchNorm2d(nChannels) # nChannels = 342
        self.fullyconnected = nn.Linear(nChannels, nClasses)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def stack_dense_layer(self, nChannels, growthRate, nDenseBlocks, bottleneck, dropRate):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck_Block(nChannels, growthRate, dropRate))
            else:
                layers.append(SingleLayer_Block(nChannels, growthRate, dropRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.convolution1(x)
        out = self.transitionblock1(self.denseblock1(out))
        out = self.transitionblock2(self.denseblock2(out))
        out = self.denseblock3(out)
        out = torch.squeeze(nnFunctional.avg_pool2d(nnFunctional.relu(self.batchNormalization_1(out)), 8))
        out = nnFunctional.log_softmax(self.fullyconnected(out), -1)
        return out

# END CODE HERE
