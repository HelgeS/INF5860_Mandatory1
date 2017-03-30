import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  m = len(y)
  D, K = W.shape

  h = X.dot(W)
  h -= np.max(h)
  h = np.exp(h) / np.sum(np.exp(h), axis=1, keepdims=True)

  for i in range(m):
    for k in range(K):
      loss += (k == y[i]) * np.log(h[i,k])
      dW[:,k] += X[i,:] * ((k == y[i]) - h[i,k])

  loss = -loss/m
  dW = -dW/m

#  dscores = h
#  dscores[range(m),y] -= 1
#  dscores /= m

#  dW = np.dot(X.T, dscores)

  loss += reg*np.sum(np.sum(W**2))/(2*m)
  dW += reg*np.sum(np.sum(W))/m
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  m = len(y)

  scores = X.dot(W)
  h = np.exp(scores - np.max(scores, axis=1, keepdims=True))
  h /= np.sum(h, axis=1, keepdims=True)

  loss = -np.sum(np.log(h[range(m),y]))/m

  dscores = h.copy()
  dscores[range(m),y] -= 1
  dscores /= m

  dW = np.dot(X.T, dscores)

  loss += reg*np.sum(np.sum(W**2))/(2*m)
  dW += reg*np.sum(np.sum(W))/m
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

