import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import numpy as np

from utils.Profiling import profile_section


class BaseCPPN:
    def get_weights(self, pos):
        pass

    def get_bias(self, pos):
        pass


class SubstrateCNN(nn.Module):
    def __init__(self, layers):
        super(SubstrateCNN, self).__init__()

        self.layers = layers

    def set_weights(self, cppn:BaseCPPN):
        conv_layers = [l for l in self.layers if l.__class__ == nn.Conv2d]

        for i, conv in enumerate(conv_layers):
            f_layer = i/len(conv_layers)

            weight_dims =[conv.out_channels, conv.in_channels, *conv.kernel_size]
            pos_weights = torch.stack((torch.full(weight_dims, f_layer),
                               *torch.meshgrid([torch.linspace( 0, 1, weight_dims[0]),
                                                torch.linspace( 0, 1, weight_dims[1]),
                                                torch.linspace(-1, 1, weight_dims[2]),
                                                torch.linspace(-1, 1, weight_dims[3])])))

            pos_weights_list = pos_weights.view(5, -1).transpose(0, 1)
            with profile_section('get_weights'):
                weights_list = cppn.get_weights(pos_weights_list)
            weights = weights_list.view(weight_dims)
            conv.weight.data = weights

            bias_dims = [conv.out_channels]
            pos_bias = torch.stack((torch.full(bias_dims, f_layer),
                                    *torch.meshgrid([torch.linspace(0, 1, bias_dims[0])])))
            pos_bias_list = pos_bias.view(2, -1).transpose(0, 1)
            with profile_section('get_biases'):
                bias_list = cppn.get_bias(pos_bias_list)
            conv.bias.data = bias_list.view(bias_dims)

    def count_all_params(self):
        return sum([p.numel() for p in self.layers.parameters()])
    def count_params(self):
        return [sum([p.numel() for p in l.parameters()]) for l in self.layers]


    def forward(self, x):
        with torch.no_grad():
            return self.layers(x)


