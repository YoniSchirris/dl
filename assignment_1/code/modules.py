"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """

    def __init__(self, in_features, out_features):
        """
        Initializes the parameters of the module.

        Args:
          in_features: size of each input sample
          out_features: size of each output sample

        TODO:
        Initialize weights self.params['weight'] using normal distribution with mean = 0 and
        std = 0.0001. Initialize biases self.params['bias'] with 0.

        Also, initialize gradients with zeros.
        """

        #######################
        # YOUR CODE HERE      #
        #######################

        # mean for weight initialization
        mu = 0

        # sigma for weight initialization
        #  this needs to be increased (e.g. to 0.001) if we want to use several layers
        sigma = 0.0001

        # initialize weights and its gradient
        weights = np.random.normal(mu, sigma, [out_features, in_features])
        weights_grad = np.zeros([out_features, in_features])

        # initialize biases and its gradients
        biases = np.zeros((out_features, 1))
        biases_grad = np.zeros((1, out_features))

        # save them for later use
        self.params = {'weight': weights, 'bias': biases}
        self.grads = {'weight': weights_grad, 'bias': biases_grad}

        ########################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
        """

        ########################
        # PUT YOUR CODE HERE  #

        # simple forward as given in the assignment
        out = np.dot(self.params['weight'], x.T)
        out += self.params['bias']
        self.x = x

        # END OF YOUR CODE    #
        #######################

        return out.T

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """

        ########################
        # PUT YOUR CODE HERE  #

        # backward as written in the paper
        W = self.params['weight']
        dx = np.dot(dout, W)

        dLdb = np.sum(dout, axis=0)

        # needs to be reshaped from [batch,] to [batch,1] for later calculations
        dLdb = np.reshape(dLdb, [np.shape(dLdb)[0], 1])
        self.grads['bias'] = dLdb

        dLdW = np.dot(dout.T, self.x)
        self.grads['weight'] = dLdW

        # END OF YOUR CODE    #
        #######################

        return dx


class ReLUModule(object):
    """
    ReLU activation module.
    """

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
        """

        ########################
        # PUT YOUR CODE HERE  #

        # simple relu
        self.I_tilde = (x > 0)

        out = x * self.I_tilde

        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        ########################
        # PUT YOUR CODE HERE  #

        # a simple multiplication
        I_tilde = self.I_tilde

        dx = dout * I_tilde

        # END OF YOUR CODE    #
        #######################

        return dx


class SoftMaxModule(object):
    """
    Softmax activation module.
    """

    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
        """

        ########################
        # PUT YOUR CODE HERE  #

        # as taken from https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
        y = np.exp((x.T - np.max(x, axis=1)).T)
        out = (y.T / np.sum(y, axis=1)).T
        self.out = out

        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        ########################
        # PUT YOUR CODE HERE  #
        xN = self.out

        # the following technique is implemented by looking at the following source:
        # https://stackoverflow.com/questions/26511401/numpy-fastest-way-of-computing-diagonal-for-each-row-of-a-2d-array

        # this creates the batch_size * n * n matrix that holds batch times diagonals
        diag_holder = np.zeros((xN.shape[0], xN.shape[1], xN.shape[1]))
        diag = np.arange(xN.shape[1])
        diag_holder[:, diag, diag] = xN

        # Here we then perform the outer product between the output of the layer
        # and subtract that from the diagonals
        dxdtilde = diag_holder - np.einsum('ij, ik -> ijk', xN, xN)

        # And we multiply this with the gradients
        dx = np.einsum('ij, ijk -> ik', dout, dxdtilde)

        # END OF YOUR CODE    #
        #######################

        return dx


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """

    def forward(self, x, y):
        """
        Forward pass.

        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss

        TODO:
        Implement forward pass of the module.
        """

        ########################
        # PUT YOUR CODE HERE  #

        out = np.mean(np.sum(y * -1 * np.log(x), axis=1))

        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, x, y):
        """
        Backward pass.

        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.

        TODO:
        Implement backward pass of the module.
        """

        ########################
        # PUT YOUR CODE HERE  #

        dx = (-np.divide(y, x)) / np.shape(y)[0]

        # END OF YOUR CODE    #
        #######################

        return dx
