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

import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch

import torch.optim as optim
from torch.utils.data import DataLoader

from part2.dataset import TextDataset
from part2.model import TextGenerationModel


################################################################################

def calculate_accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      labels: 2D int array of size [batch_size, n_classes]
              with one-hot encoding. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch

    TODO:
    Implement accuracy computation.
    """

    ########################
    # PUT YOUR CODE HERE  #

    #TODO FIX ACCURACY CALCULATION

    max_index_p = predictions.argmax(dim=2)
    max_index_t = targets
    accuracy = (max_index_p == max_index_t).float().mean().data.item()

    # END OF YOUR CODE    #
    #######################

    return accuracy


def train(config):
    # Initialize the device which to run the model on
    # device = torch.device(config.device)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = TextDataset(filename=config.txt_file, seq_length=config.seq_length)  # fixme
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    VOCAB_SIZE = dataset.vocab_size
    CHAR2IDX = dataset._char_to_ix
    IDX2CHAR = dataset._ix_to_char

    # Initialize the model that we are going to use
    model = TextGenerationModel(batch_size=config.batch_size,
                                seq_length=config.seq_length,
                                vocabulary_size=VOCAB_SIZE,  # fixme
                                lstm_num_hidden=config.lstm_num_hidden,
                                lstm_num_layers=config.lstm_num_layers,
                                device=device)




    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    #TODO Define the total loss as average of cross-entropy loss over all timesteps (Equation 13).

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    if None:
        model.load_state_dict(torch.load('./intermediate-model-step-400.pth'))
        optimizer.load_state_dict(torch.load("./intermediate-optim-step-400.pht"))

        print("Loaded it!")

    model.to(device)

    # TODO Which optimizer would we want to use?
    # optimizer = torch.optim.RMSProp(model.parameters(), lr=config.learning_rate)

    EPOCHS = 50

    for epoch in range(EPOCHS):
        # initialization of state that's given to the forward pass
        # reset every epoch
        h, c = model.reset_lstm(config.batch_size)
        h = h.to(device)
        c = c.to(device)

        for step, (batch_inputs, batch_targets) in enumerate(data_loader):

            # Only for time measurement of step through network
            t1 = time.time()

            model.train()

            optimizer.zero_grad()

            x = torch.stack(batch_inputs, dim=1).to(device)
            y = torch.stack(batch_targets,  dim=1).to(device)

            x = one_hot_encode(x, VOCAB_SIZE)

            output, (h, c) = model(x=x, prev_state=(h, c))

            loss = criterion(output.transpose(1, 2), y)  # fixme -- might need some dimension fixing

            accuracy = calculate_accuracy(output, y)  # fixme -- might need some dimension fixing
            h = h.detach()
            c = c.detach()
            loss.backward()
            # add clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
            optimizer.step()

            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size / float(t2 - t1)

            if step % config.print_every == 0:
                #TODO FIX THIS PRINTING
                print(f"Epoch {epoch} Train Step {step}/{config.train_steps}, Examples/Sec = {examples_per_second}, Accuracy = {accuracy}, Loss = {loss}")
                #
                # print("[{}]".format(datetime.now().strftime("%Y-%m-%d %H:%M")))
                # print("[{}] Train Step {:04f}/{:04f}, Batch Size = {}, Examples/Sec = {:.2f}, Accuracy = {:.2f}, Loss = {:.3f}".format(
                #     datetime.now().strftime("%Y-%m-%d %H:%M"), step, config.train_steps, config.batch_size, examples_per_second, accuracy, loss
                # ))

                # print(loss)


            if step % config.sample_every == 0:
                FIRST_CHAR = 'I'  # fixme should this be randomized?
                predict(device, model, FIRST_CHAR, VOCAB_SIZE, IDX2CHAR, CHAR2IDX)
                # Generate some sentences by sampling from the model
                path_model = 'intermediate-model-step-{}.pth'.format(step)
                path_optimizer = 'intermediate-optim-step-{}.pht'.format(step)
                torch.save(model.state_dict(),path_model)
                torch.save(optimizer.state_dict().path_optimizer)

            if step == config.train_steps:
                # If you receive a PyTorch data-loader error, check this bug report:
                # https://github.com/pytorch/pytorch/pull/9655
                break

    print('Done training.')


# def predict(device, model, first_char, vocab_size, idx2char, char2idx, T=30):
#
#     #
#     consider_top_characters = 1
#
#     # TODO Should we only check the PREVIOUS character, or ALL previous characters?
#
#     output = first_char
#     model.eval()
#
#     h, c = model.reset_lstm(1)
#     h = h.to(device)
#     c = c.to(device)
#
#     output_sentence = first_char
#
#     for character_num in range(T):
#         # THis currently only checks the last character
#
#         idx = torch.tensor(char2idx[output_sentence]).to(device).view(1,1)
#         one_hot_index = one_hot_encode(idx, vocab_size)
#         output, (h, c) = model(one_hot_index, (h, c))
#
#         output = idx2char[torch.topk(output[0], 1)[1].tolist()[0][0]]
#
#         # TODO Add temperature
#         # output = np.random.choice(torch.topk(output[5], 1)[1].tolist()[0]) << Here we can do something with the temperature
#
#         output_sentence += output
#
#     print(output_sentence)

def predict(device, model, first_char, vocab_size, idx2char, char2idx, T=30):

    #
    consider_top_characters = 1

    # TODO Should we only check the PREVIOUS character, or ALL previous characters?

    output = first_char
    model.eval()

    h, c = model.reset_lstm(1)
    h = h.to(device)
    c = c.to(device)

    first_char = "De"


    output_sentence = first_char
    input_sentence = []
    for char in output_sentence:
        input_sentence.append(char2idx[char])
    # THis currently only checks the last character

    idx = torch.tensor(input_sentence).to(device).view(len(input_sentence), 1)
    one_hot_index = one_hot_encode(idx, vocab_size)
    one_hot_index = one_hot_index.view(one_hot_index.size()[1], one_hot_index.size()[0], one_hot_index.size()[2])
    output, (h, c) = model(one_hot_index, (h, c))
    output = torch.topk(output[0], 1)[1].tolist()[0][0]
    output_sentence += idx2char[output]

    for character_num in range(T):

        idx = torch.tensor(output).to(device).view(1, 1)
        one_hot_index = one_hot_encode(idx, vocab_size)
        one_hot_index = one_hot_index.view(one_hot_index.size()[1], one_hot_index.size()[0], one_hot_index.size()[2])
        output, (h, c) = model(one_hot_index, (h, c))

        output = torch.topk(output[0], 1)[1].tolist()[0][0]

        # TODO Add temperature
        # output = np.random.choice(torch.topk(output[5], 1)[1].tolist()[0]) << Here we can do something with the temperature

        output_sentence += idx2char[output]


    print(output_sentence)


def one_hot_encode(input, vocab_size):
    one_hot_size = list(input.size())

    one_hot_size.append(vocab_size)
    output = torch.zeros(one_hot_size).to(input.device)
    output.scatter_(2, input.unsqueeze(-1), 1)
    return output



################################################################################
################################################################################

if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1e6, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')

    config = parser.parse_args()

    # Train the model
    train(config)
