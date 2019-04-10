"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

import torch.nn as nn
from torch.nn import functional as F



class ConvNet(nn.Module):
    """
    This class implements a Convolutional Neural Network in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an ConvNet object can perform forward.
    """

    def __init__(self, n_channels, n_classes):
        """
        Initializes ConvNet object.

        Args:
          n_channels: number of input channels
          n_classes: number of classes of the classification problem


        TODO:
        Implement initialization of the network.
        """

        ########################
        # PUT YOUR CODE HERE  #\

        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0,
        # dilation=1, groups=1, bias=True)

        # torch.nn.MaxPool2d(kernel_size, stride=None, padding=0,
        #                    dilation=1, return_indices=False, ceil_mode=False)

        # avg pool:
        # kernel_size: Any,
        # stride: Any = None,
        # padding: int = 0,
        # ceil_mode: bool = False,
        # count_include_pad: bool = True)

        super(ConvNet, self).__init__()

        # I assume we keep the std parameters for conv2d as dilation=1, groups=1, bias=True
        # I assume we keep the std parameters for maxpool as dilation=1, return_indices=false, ceil_mode=false

        self.feats = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            # maxpool1
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # conv2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            # maxpool2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # conv3_a
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),

            # conv3_b
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            # maxpool3
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # conv4_a
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            # conv4_b
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            # maxpool4
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # conv5_a
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            # conv5_b
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            # maxpool5
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # avgpool
            nn.AvgPool2d(kernel_size=1, stride=1, padding=0),


        )

        self.linearLayer = nn.Linear(in_features=512, out_features=10)

        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network

        TODO:
        Implement forward pass of the network.
        """

        ########################
        # PUT YOUR CODE HERE  #

        out = self.feats(x)

        # linear
        out = self.linearLayer(out.view(x.shape[0], -1))

        # END OF YOUR CODE    #
        #######################

        return out
