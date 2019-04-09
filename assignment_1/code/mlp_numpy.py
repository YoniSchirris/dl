"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import * 

class MLP(object):
  """
  This class implements a Multi-layer Perceptron in NumPy.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward and backward.
  """

  def __init__(self, n_inputs, n_hidden, n_classes):
    """
    Initializes MLP object. 
    
    Args:
      n_inputs: number of inputs.
      n_hidden: list of ints, specifies the number of units
                in each linear layer. If the list is empty, the MLP
                will not have any linear layers, and the model
                will simply perform a multinomial logistic regression.
      n_classes: number of classes of the classification problem.
                 This number is required in order to specify the
                 output dimensions of the MLP
    
    TODO:
    Implement initialization of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #

    self.layers = []

    # add the first layer, it takes n_inputs and outputs the number for the first linear layer
    self.layers.append(LinearModule(n_inputs, n_hidden[0]))

    # add the relu activation for first layer
    self.layers.append(ReLUModule())

    # for each of the elements up until the first-to-last, add a linear layer with relu activation
    for i in range(len(n_hidden)-1):
          # here, input is output from previous
          self.layers.append(LinearModule(n_hidden[i], n_hidden[i+1]))
          self.layers.append(ReLUModule())

    # finally, we want the final layer to output the number of classes
    self.layers.append(LinearModule(n_hidden[-1], n_classes))

    # and the get the softmax
    self.layers.append(SoftMaxModule())
    

    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    
    TODO:
    Implement forward pass of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #

    for layer in self.layers:
          x = layer.forward(x)

    out = x
    # NUMBER_OF_LINEAR_LAYERS = len(self.linears)

    # linear_out = self.linears[0].forward(x)
    # relu_out = self.relu.forward(linear_out)

    # if NUMBER_OF_LINEAR_LAYERS > 1:
    #   for i in range(1, NUMBER_OF_LINEAR_LAYERS-1):
    #     linear_out = self.linears[i].forward(relu_out)
    #     relu_out = self.relu.forward(linear_out)
    
    # linear_out = self.linears[-1].forward(relu_out)

    # softmax_out = self.softmax.forward(relu_out)
               
    # out = softmax_out
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Performs backward pass given the gradients of the loss. 

    Args:
      dout: gradients of the loss
    
    TODO:
    Implement backward pass of the network.
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    dx = dout

    for layer in self.layers[::-1]:
          dx = layer.backward(dx)

    # END OF YOUR CODE    #
    #######################

    return
