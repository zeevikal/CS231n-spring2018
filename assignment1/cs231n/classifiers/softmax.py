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
  scores = X.dot(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]

  # Softmax Loss
  for i in range(num_train):
    f = scores[i] - np.max(scores[i])
    softmax = np.exp(f)/np.sum(np.exp(f))
    loss += -np.log(softmax[y[i]])
    # Weight Gradients
    for j in range(num_classes):
      dW[:,j] += X[i] * softmax[j]
    dW[:,y[i]] -= X[i]

  # Average
  loss /= num_train
  dW /= num_train

  # Regularization
  loss += reg * np.sum(W * W)
  dW += reg * 2 * W 
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
  num_train = X.shape[0]
  
  # scores have number of examples rows and classes columns
  scores = X.dot(W)
  scores -= np.max(scores)  # trick that control number instability
  # this gives the correct class score for each example
  correct_class_score = scores[np.arange(num_train), y].reshape(-1, 1)

  exp_sum = np.sum(np.exp(scores), axis=1).reshape(-1, 1)
  loss = np.sum(- correct_class_score + np.log(exp_sum))
  
  softmax = np.exp(scores) / exp_sum
  grad = softmax
  grad[np.arange(num_train), y] -= 1
  dW = X.T.dot(grad)
  loss /= num_train
  dW /= num_train
  loss += reg * np.sum(W * W)
  dW += reg * 2 * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

