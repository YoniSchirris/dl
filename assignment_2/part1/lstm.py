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

class LSTM(nn.Module):
    #TODO NOTE THAT LR NEEDS TO BE 0.01 FOR THE LSTM TO WORK

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(LSTM, self).__init__()

        self.seq_length = seq_length
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.device = device

        self.sigma = sigma = 1

        # I took some inspiration from:
        # https://github.com/keitakurita/Practical_NLP_in_PyTorch/blob/master/deep_dives/lstm_from_scratch.ipynb
        # g
        self.Wgx = nn.Parameter(sigma * torch.randn((input_dim, num_hidden), device=device))
        self.Wgh = nn.Parameter(sigma * torch.randn((num_hidden, num_hidden), device=device))
        self.bg = nn.Parameter(torch.zeros(num_hidden, device=device))


        # i
        self.Wix = nn.Parameter(sigma * torch.randn((input_dim, num_hidden), device=device))
        self.Wih = nn.Parameter(sigma * torch.randn((num_hidden, num_hidden), device=device))
        self.bi = nn.Parameter(torch.zeros(num_hidden, device=device))

        # f
        self.Wfx = nn.Parameter(sigma * torch.randn((input_dim, num_hidden), device=device))
        self.Wfh = nn.Parameter(sigma * torch.randn((num_hidden, num_hidden), device=device))
        self.bf = nn.Parameter(torch.zeros(num_hidden, device=device))
        # o
        self.Wox = nn.Parameter(sigma * torch.randn((input_dim, num_hidden), device=device))
        self.Woh = nn.Parameter(sigma * torch.randn((num_hidden, num_hidden), device=device))
        self.bo = nn.Parameter(torch.zeros(num_hidden, device=device))

        # linear
        self.Wph = nn.Parameter(sigma * torch.randn((num_hidden, num_classes), device=device))
        self.bp = nn.Parameter(torch.zeros(num_classes, device=device))

        # hidden and cell

        self.h = torch.zeros(self.num_hidden).to(device)
        self.c = torch.zeros(self.num_hidden).to(device)


    def reset_lstm(self):

        # we reset the cell state and hidden state here

        self.c.detach_()

        # reset cell state to all zeros
        nn.init.constant_(self.c, 0.0)

        self.h.detach_()

        # reset hidden state to all zeros
        nn.init.constant_(self.h, 0.0)


    def forward(self, x):

        self.reset_lstm()

        for t in range(self.seq_length):
            x_t = x[:, t].view(self.batch_size, self.input_dim)
            i_t = torch.sigmoid(x_t @ self.Wix + self.h @ self.Wih + self.bi)
            f_t = torch.sigmoid(x_t @ self.Wfx + self.h @ self.Wfh + self.bf)
            g_t = torch.tanh(x_t @ self.Wgx + self.h @ self.Wgh + self.bg)
            o_t = torch.sigmoid(x_t @ self.Wox + self.h @ self.Woh + self.bo)

            self.c = g_t * i_t + self.c * f_t

            self.h = torch.tanh(self.c) * o_t

        p = self.h @ self.Wph + self.bp

        return p
