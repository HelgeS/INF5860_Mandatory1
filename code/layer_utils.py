from code.layers import *


def affine_relu_forward(x, w, b):
  """
  Convenience layer that perorms an affine transform followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  out, cache = None, None
  #############################################################################
  # TODO: Combind Relu and affine forward, into a convenience function        #
  #############################################################################
  a, fw_cache = affine_forward(x, w, b)
  out, relu_cache = relu_forward(a)
  cache = (fw_cache, relu_cache)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return out, cache


def affine_relu_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Combind Relu and affine backward, into a convenience function       #
  #############################################################################
  fw_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = affine_backward(da, fw_cache)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db
