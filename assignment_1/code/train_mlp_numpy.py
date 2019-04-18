"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100

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

    batch_size = np.shape(predictions)[0]

    predictions = (predictions == predictions.max(axis=1)[:, None]).astype(int)

    correct_predictions = predictions * targets
    accuracy = np.mean(np.sum(correct_predictions, axis=1))

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

    ## Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []

    ########################
    # PUT YOUR CODE HERE  #


    # set all flag variables, except dor dnn_hidden_units
    LEARNING_RATE_DEFAULT = FLAGS.learning_rate
    MAX_STEPS_DEFAULT = FLAGS.max_steps
    BATCH_SIZE_DEFAULT = FLAGS.batch_size
    EVAL_FREQ_DEFAULT = FLAGS.eval_freq

    # get test data to initialize the model with
    cifar10 = cifar10_utils.get_cifar10(DATA_DIR_DEFAULT)

    x_test, y_test = cifar10['test'].images, cifar10['test'].labels

    # input_size is a concatenation of the 3d image
    input_size = np.shape(x_test)[1] * np.shape(x_test)[2] * np.shape(x_test)[3]

    # this is the class size = output size
    class_size = np.shape(y_test)[1]

    # reshape the images to be 1-dimensional vectors
    x_test = x_test.reshape([np.shape(x_test)[0], input_size])

    model = MLP(n_inputs=input_size, n_hidden=dnn_hidden_units, n_classes=class_size)

    calculate_loss = CrossEntropyModule()

    # will store are results
    accuracies_val = []
    accuracies_train = []
    loss_val = []
    loss_train = []

    # keep going until we reach the max steps
    for step in range(MAX_STEPS_DEFAULT):

        # get the next batch
        x, y = cifar10['train'].next_batch(BATCH_SIZE_DEFAULT)

        x = x.reshape([np.shape(x)[0], input_size])

        forward_out = model.forward(x)

        loss = calculate_loss.forward(forward_out, y)

        loss_gradient = calculate_loss.backward(forward_out, y)

        model.backward(loss_gradient)

        for layer in model.layers:
            if hasattr(layer, 'params'):
                # this only holds for the linear layers
                # update weights and bias
                layer.params['weight'] = layer.params['weight'] - LEARNING_RATE_DEFAULT * layer.grads['weight']
                layer.params['bias'] = layer.params['bias'] - LEARNING_RATE_DEFAULT * layer.grads['bias']

        # evaluate every EVAL_FREQ_DEFAULT steps
        if step % EVAL_FREQ_DEFAULT == 0:
            test_forward = model.forward(x_test)
            accuracies_train.append(accuracy(forward_out, y))
            accuracies_val.append(accuracy(test_forward, y_test))
            loss_train.append(loss)
            loss_val.append(calculate_loss.forward(test_forward, y_test))


    print("accuracies train")
    print(accuracies_train)
    print("accuracies val")
    print(accuracies_val)
    print("losses train")
    print(loss_train)
    print("losses val")
    print(loss_val)


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
    FLAGS, unparsed = parser.parse_known_args()

    main()
