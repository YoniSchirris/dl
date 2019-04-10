"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils

import torch
from torch import optim

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None


def accuracy(predictions, targets):
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
    max_index_p = predictions.argmax(dim=1)
    max_index_t = targets.argmax(dim=1)
    accuracy = (max_index_p == max_index_t).float().mean().data.item()
    # END OF YOUR CODE    #
    #######################

    return accuracy


def train():
    """
    Performs training and evaluation of ConvNet model.

    TODO:
    Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
    """

    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)

    ########################
    # PUT YOUR CODE HERE  #


    # check if GPU is available. If not, use CPU
    if torch.cuda.is_available():
        torch.device('cuda')
    else:
        torch.device('cpu')

    cifar10 = cifar10_utils.get_cifar10(DATA_DIR_DEFAULT)

    x_test, y_test = cifar10['test'].images, cifar10['test'].labels

    num_channels = np.shape(x_test)[1]
    class_size = np.shape(y_test)[1]

    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)

    net = ConvNet(n_channels=num_channels, n_classes=class_size)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE_DEFAULT)

    for step in range(MAX_STEPS_DEFAULT):
        print(step)
        x, y = cifar10['train'].next_batch(BATCH_SIZE_DEFAULT)
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        optimizer.zero_grad()

        out = net(x)

        y = y.argmax(dim=1)

        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        print(loss.item())

        if step % EVAL_FREQ_DEFAULT == 0:
            # test_out = net.forward(x_test)
            # print(accuracy(test_out, y_test))
            test_out = net.forward(x)
            print(accuracy(test_out, y))


# END OF YOUR CODE    #
    #######################


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():
    """
    Main function
    """
    # Print all Flags to confirm parameter settings
    print_flags()

    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)

    # Run the training operation
    train()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()

    main()
