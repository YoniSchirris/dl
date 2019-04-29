# MIT License
#
# Copyright (c) 2017 Tom Runia
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


class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0'):

        super(TextGenerationModel, self).__init__()
        # Initialization here...
        # I got inspiration from https://machinetalk.org/2019/02/08/text-generation-with-pytorch/

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocabulary_size = vocabulary_size
        self.lstm_num_hidden = lstm_num_hidden
        self.lstm_num_layers = lstm_num_layers
        self.device = device

        # this needed in our case?
        # self.embedding = nn.Embedding(vocabulary_size, vocabulary_size)

        self.lstm = nn.LSTM(input_size=vocabulary_size,
                            hidden_size=lstm_num_hidden,
                            num_layers=lstm_num_layers,
                            batch_first=True)
        self.dense = nn.Linear(lstm_num_hidden, vocabulary_size, bias=True)



    def forward(self, x, prev_state=None):

        # if prev_state == None:
        #     (h, c) = self.reset_lstm(1)
        # Implementation here...
        # embed = self.embedding(x)  # FIXME this required?
        # embed = embed.view((self.seq_length, self.batch_size, self.vocabulary_size))
        output, state = self.lstm(x, prev_state)  # fixme embed needs to be three-dimensional: (seq_length, batch, input_size)

        out = self.dense(output)

        return out, state


    # used when doing more than 1 epoch to reset.
    def reset_lstm(self, batch_size):
        # similar to reset_lstm in part1/lstm.py, but used differently as we're using torch modules here
        return (torch.zeros(self.lstm_num_layers, batch_size, self.lstm_num_hidden),
                torch.zeros(self.lstm_num_layers, batch_size, self.lstm_num_hidden))
