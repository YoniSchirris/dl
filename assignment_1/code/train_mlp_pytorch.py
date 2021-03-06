"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils

import torch
import torch.optim as optim

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100


# self-added variables
OPTIMIZER_DEFAULT = 'sgd'
REGULARIZER_DEFAULT = 0
MOMENTUM_DEFAULT = 0

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
    Performs training and evaluation of MLP model.

    TODO:
    Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
    """

    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)


    ## Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []



    ########################
    # PUT YOUR CODE HERE  #

    # because I don't have a GPU and the training was quick enough on a CPU,
    # I don't save my tensor on a GPU

    LEARNING_RATE_DEFAULT = FLAGS.learning_rate
    MAX_STEPS_DEFAULT = FLAGS.max_steps
    BATCH_SIZE_DEFAULT = FLAGS.batch_size
    EVAL_FREQ_DEFAULT = FLAGS.eval_freq
    OPTIMIZER_DEFAULT = FLAGS.optimizer

    # self-added variables
    REGULARIZER_DEFAULT = FLAGS.regularizer
    MOMENTUM_DEFAULT = FLAGS.momentum

    # get test data to initialize the model with
    cifar10 = cifar10_utils.get_cifar10(DATA_DIR_DEFAULT)

    x_test, y_test = cifar10['test'].images, cifar10['test'].labels

    input_size = np.shape(x_test)[1] * np.shape(x_test)[2] * np.shape(x_test)[3]
    class_size = np.shape(y_test)[1]

    x_test = torch.from_numpy(x_test.reshape([np.shape(x_test)[0], input_size]))
    y_test = torch.from_numpy(y_test)

    net = MLP(n_inputs=input_size, n_hidden=dnn_hidden_units, n_classes=class_size)

    criterion = torch.nn.CrossEntropyLoss()

    eval_accuracies = []
    train_accuracies = []

    eval_loss = []
    train_loss = []

    # choose between optimizer
    if OPTIMIZER_DEFAULT == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE_DEFAULT, momentum=MOMENTUM_DEFAULT, weight_decay=REGULARIZER_DEFAULT)
    elif OPTIMIZER_DEFAULT == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE_DEFAULT, weight_decay=REGULARIZER_DEFAULT)

    for step in range(MAX_STEPS_DEFAULT):
        x, y = cifar10['train'].next_batch(BATCH_SIZE_DEFAULT)
        x = x.reshape([np.shape(x)[0], input_size])
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        optimizer.zero_grad()

        out = net.forward(x)
        # convert out and y to index of max (class prediction)?


        # required?
        # x = x.argmax(dim=1)

        loss = criterion(out, y.argmax(dim=1))
        loss.backward()
        optimizer.step()
        # print(loss.item())

        if step % EVAL_FREQ_DEFAULT == 0:

            test_out = net.forward(x_test)
            # print(accuracy(test_out, y_test))
            eval_accuracies.append(accuracy(test_out, y_test))
            train_accuracies.append(accuracy(out, y))

            eval_loss.append(criterion(test_out, y_test.argmax(dim=1)).data.item())
            train_loss.append(criterion(out, y.argmax(dim=1)).data.item())
    # final accuracy calculation

    test_out = net.forward(x_test)
    print("EVAL ACCURACY")
    print(eval_accuracies)
    print("train ACCURACY")
    print(train_accuracies)
    print("EVAL loss")
    print(eval_loss)
    print("train loss")
    print(train_loss)

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
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
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
    # SELF ADDED CONSTANTS
    parser.add_argument('--optimizer', type=str, default=OPTIMIZER_DEFAULT,
                        help='adam or sgd')
    parser.add_argument('--regularizer', type=float, default=REGULARIZER_DEFAULT,
                        help='Weight decay for the adam optimizer')
    parser.add_argument('--momentum', type=float, default=MOMENTUM_DEFAULT,
                        help='Momentum for the SGD optimizer')


    FLAGS, unparsed = parser.parse_known_args()

    main()
