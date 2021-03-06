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

    # only returns the actual floats, not the tensors
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

    # save input given via bash script
    LEARNING_RATE_DEFAULT = FLAGS.learning_rate
    MAX_STEPS_DEFAULT = FLAGS.max_steps
    BATCH_SIZE_DEFAULT = FLAGS.batch_size
    EVAL_FREQ_DEFAULT = FLAGS.eval_freq

    # lists to save all the results
    accuracies_on_train = []
    accuracies_on_test = []
    trainLosses = []
    valLosses = []

    # check if GPU is available. If not, use CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    cifar10 = cifar10_utils.get_cifar10(DATA_DIR_DEFAULT)

    # because RGB images
    num_channels = 3
    class_size = 10

    # get iterations required for test batching for accuracy calculation
    # this is required to save GPU memory. Or maybe it isn't, because the memory issues I had
    # were because of saving tensors.
    # anyway...
    num_test_iters = int(np.ceil(cifar10['test']._num_examples) / BATCH_SIZE_DEFAULT)

    # initialize model and set it to GPU if it's there
    net = ConvNet(n_channels=num_channels, n_classes=class_size).to(device)

    # initialize loss function
    criterion = torch.nn.CrossEntropyLoss()

    # initialize adam optimizer
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE_DEFAULT)

    for step in range(MAX_STEPS_DEFAULT):
        # as a check to see if it runs correctly, I run the steps where it is
        print(step)

        # get the data
        x, y = cifar10['train'].next_batch(BATCH_SIZE_DEFAULT)

        # set it to the GPU if it's there
        x = torch.tensor(x).to(device)
        y = torch.tensor(y).to(device)

        # reset gradients
        optimizer.zero_grad()

        # get the output from the next
        out = net(x)

        # get the loss
        trainLoss = criterion(out, y.argmax(dim=1))

        # run backprop
        # <3 pytorch
        trainLoss.backward()
        optimizer.step()

        # detach to save some memory
        out.detach()

        # every eval_freq_default steps...
        if step % EVAL_FREQ_DEFAULT == 0:
            # intermediate saving.. of results
            accuracies_on_test_intermediate = []
            loss_on_test_intermediate = []

            # for all itereations required to go through the entire test data given the batch size
            for test_iter in range(num_test_iters):
                # get data
                x_test, y_test = cifar10['test'].next_batch(BATCH_SIZE_DEFAULT)
                # set to gpu
                x_test = torch.tensor(x_test, requires_grad=False).to(device)
                y_test = torch.tensor(y_test, requires_grad=False).to(device)

                # get output
                test_out = net(x_test)

                # calculate accuracies and losses
                accuracies_on_test_intermediate.append(accuracy(test_out, y_test))
                # don't save tensors, only the results!
                loss_on_test_intermediate.append(criterion(test_out, y_test.argmax(dim=1)).data.item())

                # save some memory
                test_out.detach()
                x_test.detach()
                y_test.detach()

            # valLoss = criterion(test_out, y_test.argmax(dim=1))

            # mean over all intermediate results
            valLoss = np.mean(loss_on_test_intermediate)
            valAccuracy = np.mean(accuracies_on_test_intermediate)
            trainAccuracy = accuracy(out, y)

            # and save
            trainLosses.append(trainLoss.data.item())
            valLosses.append(valLoss)
            accuracies_on_test.append(valAccuracy)
            accuracies_on_train.append(trainAccuracy)


    # print all results
    print("accuracy on train")
    print(accuracies_on_train)
    print("accuracy on test")
    print(accuracies_on_test)
    print("validation losses")
    print(valLosses)
    print("train losses")
    print(trainLosses)


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
