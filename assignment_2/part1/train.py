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

import argparse
import time
from datetime import datetime
import numpy as np

import torch
from torch.utils.data import DataLoader

from part1.dataset import PalindromeDataset
from part1.vanilla_rnn import VanillaRNN
from part1.lstm import LSTM

from torch.autograd import Variable


# You may want to look into tensorboardX for logging
# from tensorboardX import SummaryWriter

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

    # same as assignment 1
    max_index_p = predictions.argmax(dim=1)
    max_index_t = targets
    accuracy = (max_index_p == max_index_t).float().mean().data.item()

    # END OF YOUR CODE    #
    #######################

    return accuracy

def train(config):
    assert config.model_type in ('RNN', 'LSTM')

    # Initialize the device which to run the model on
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # set the config params to be used here
    SEQ_LENGTH = config.input_length
    INPUT_DIM = config.input_dim
    NUM_HIDDEN = config.num_hidden
    NUM_CLASSES = config.num_classes
    BATCH_SIZE = config.batch_size

    # Initialize the model that we are going to use
    if config.model_type == "RNN":
        model = VanillaRNN(seq_length=SEQ_LENGTH, input_dim=INPUT_DIM, num_hidden=NUM_HIDDEN, num_classes=NUM_CLASSES,
                       batch_size=BATCH_SIZE, device=device)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)

    elif config.model_type == "LSTM":
        model = LSTM(seq_length=SEQ_LENGTH, input_dim=INPUT_DIM, num_hidden=NUM_HIDDEN, num_classes=NUM_CLASSES,
                       batch_size=BATCH_SIZE, device=device)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate, momentum=0.8, weight_decay=1e-4)

    model.to(device)
    # Initialize the dataset and data loader (note the +1)
    dataset = PalindromeDataset(config.input_length + 1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Setup the loss and optimizer (optimize done above)
    criterion = torch.nn.CrossEntropyLoss()

    # for intermediate reporting and convergence checks
    intermediate_accuracies = []

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        t1 = time.time()

        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)

        ############################################################################
        # QUESTION: what happens here and why?
        # ANSWER: the gradients are clipped /rescaled to a max value, as explained in slide 50 of lecture 6
        ############################################################################
        ############################################################################

        out = model.forward(batch_inputs)
        loss = criterion(out, batch_targets)

        optimizer.zero_grad()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
        optimizer.step()

        accuracy = calculate_accuracy(out, batch_targets)

        intermediate_accuracies.append(accuracy)

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size / float(t2 - t1)

        if step % 10000== 0:
            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                config.train_steps, config.batch_size, examples_per_second,
                accuracy, loss
            ))

        if step > 10:
            # check for convergence: If the last 5 measured accuracies' mean is over .98, we'll say it converges
            if step == config.train_steps or np.mean(intermediate_accuracies[-5:-1]) >= 0.98:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
                break

    print('Done training.')
    print('finally accuracy:')
    print(accuracy)
    return accuracy, loss.data.item()


################################################################################
################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=1.0)
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

    parser.add_argument('--experiment', type=str, default="0",
                        help="A string. Give 1 to run an accuracy experiment after training")

    config = parser.parse_args()

    # Experiments!
    if config.experiment == "1.3" or config.experiment == "1.6":
        # get the configuration
        print(config)
        if config.experiment == "1.3":
            # set the range of sequence lengths
            Ts = range(5, 20)
        if config.experiment == "1.6":
            Ts = range(5, 200, 5)
        # save all results
        experimental_accuracies = []
        experimental_losses = []
        for T in Ts:
            # for each sequence length
            config.input_length = T
            # more saving of intermediate intermediate results
            itmdt_acc = []
            itmdt_loss = []
            for j in range(5):
                # for each sequence length, run it 5 times
                acc, loss = train(config)
                itmdt_acc.append(acc)
                itmdt_loss.append(loss)
            # and then average this, add it to result list
            experimental_accuracies.append(np.mean(itmdt_acc))
            experimental_losses.append(np.mean(itmdt_loss))

            # intermediate printing
            print("-----")
            print("T = {}".format(T))
            print("Accuracies")
            print(experimental_accuracies)
            print("losses")
            print(experimental_losses)
            print("-----")
        # final printing
        print("------------------")
        print("FINAL ACCURACIES")
        print(experimental_accuracies)
        print("------------------")
        print("------------------")
        print("FINAL LOSSES")
        print(experimental_losses)
        print("------------------")
    else:
        # train the model normally
        train(config)
