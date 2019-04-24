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
        sigma = 1e-4

        np.random.seed(42)

        # initialize variables

        # check sizes of every single fucking parameter
        self.Whx = nn.Parameter(torch.tensor(np.random.normal(mu, sigma, size=(num_hidden, input_dim))))

        self.Whh = nn.Parameter(torch.tensor(np.random.normal(mu, sigma, size=(num_hidden, num_hidden))))

        self.Wph = nn.Parameter(torch.tensor(np.random.normal(mu, sigma, size=(num_classes, num_hidden))))

        self.bp = nn.Parameter(torch.zeros(size=(num_classes, 1))).double()

        self.bh = nn.Parameter(torch.zeros(size=(num_hidden, 1))).double()

        self.h = torch.zeros((num_hidden, 1), requires_grad=False)

        self.seq_length = seq_length

    def forward(self, x):
        h = self.h
        for seq_idx in range(self.seq_length):
            a = self.Whx @ x[:, seq_idx].view(1, -1).double()
            b = self.Whh.double() @ h.double()
            c = self.bh
            h = torch.tanh(a + b + c)


        p = (self.Wph @ h) + self.bp
        return p.t()
