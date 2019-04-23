################################################################################
# MIT License
#
# Copyright (c) 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

import numpy as np


################################################################################

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(VanillaRNN, self).__init__()

        mu = 0
        sigma = 1e-3

        np.random.seed(42)

        # initialize variables
        self.Whx = nn.Parameter(torch.tensor(np.random.normal(mu, sigma, size=(seq_length, num_hidden))))

        self.Whh = nn.Parameter(torch.tensor(np.random.normal(mu, sigma, size=(num_hidden, num_hidden))))

        self.Wph = nn.Parameter(torch.tensor(np.random.normal(mu, sigma, size=(num_hidden, num_classes))))

        self.bp = nn.Parameter(torch.zeros(size=(batch_size, num_classes))).double()

        self.bh = nn.Parameter(torch.zeros(size=(batch_size, num_hidden))).double()

        self.h = torch.zeros((batch_size, num_hidden))

        self.seq_length = seq_length

        self.num_hidden = num_hidden

    def forward(self, x):
        for seq in range(self.seq_length):
            a = x.double()@self.Whx.double()
            b = self.h.double()@self.Whh.double()
            c = self.bh
            self.h = torch.tanh(a + b + c)

        p = self.h@self.Wph + self.bp
        return p
