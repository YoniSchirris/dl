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
    
    ########################
    mu = 0
    sigma = 0.0001
    
    # weights are R ^ out * in
    weights = np.random.normal(mu, sigma, [in_features, out_features])
    weights_grad = np.zeros([in_features, out_features])

    # biases are R ^ out
    biases = np.zeros(out_features)
    biases_grad = np.zeros(out_features)    
    #######################
    
    self.params = {'weight': weights, 'bias': biases}
    self.grads = {'weight': weights_grad, 'bias': biases_grad}

    #raise NotImplementedError
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

    # W [out * in ] * x [in] + b [ out ]


    out = np.dot(x, self.params['weight'])
    out += self.params['bias']

    self.x = x
    # END OF YOUR CODE    #
    #######################

    return out

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
    
    W = self.params['weight']
    dx = np.dot(dout, W.T)

    # dx = np.einsum('k ,ij ->',dout, W)
    dLdb = np.mean(dout, axis=0)

    self.params['bias'] += dLdb

    # average over the batch now
    dLdW = np.mean((np.dot(self.x.T, dx)), axis=1)


    

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
    
    # x[x<0]=0
    #  out = x

    self.I_tilde = (x>0)

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
    y = np.exp(x - np.max(x))
    out = y / np.sum(y)
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

    # https://stackoverflow.com/questions/26511401/numpy-fastest-way-of-computing-diagonal-for-each-row-of-a-2d-array
    diag_holder = np.zeros((xN.shape[0], xN.shape[1], xN.shape[1]))
    diag = np.arange(xN.shape[1])
    diag_holder[:, diag, diag] = xN

    dxdtilde = diag_holder - np.einsum('ij, ik -> ijk', xN, xN)

    dx = np.einsum('ij, ijk -> ij', dout, dxdtilde)
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
    
    out = -1 * np.shape(np.sum(x*y, axis=1))

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
    
    dx = -1 * np.divide(y, x)

    # END OF YOUR CODE    #
    #######################

    return dx
