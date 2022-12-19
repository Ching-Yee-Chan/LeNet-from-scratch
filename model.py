import numpy as np
from layers import *


class LeNet5:
    """
    Lenet 5 has the following architecture:

    conv - relu - 2x2 max pool - conv - relu - 2x2 max pool - affine - relu - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(
        self,
        input_dim=(1, 32, 32),
        num_filters=(6, 16),
        filter_size=5,
        hidden_dim=(16*5*5, 120, 84),
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=np.float32,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        C, H, W = input_dim
        self.params['W1'] = np.random.normal(scale=weight_scale, size=(num_filters[0], C, filter_size, filter_size))
        self.params['b1'] = np.zeros((num_filters[0]))
        self.params['W2'] = np.random.normal(scale=weight_scale, size=(num_filters[1], num_filters[0], filter_size, filter_size))
        self.params['b2'] = np.zeros((num_filters[1]))
        self.params['W3'] = np.random.normal(scale=weight_scale, size=(hidden_dim[0] , hidden_dim[1]))
        self.params['b3'] = np.zeros((hidden_dim[1]))
        self.params['W4'] = np.random.normal(scale=weight_scale, size=(hidden_dim[1], hidden_dim[2]))
        self.params['b4'] = np.zeros((hidden_dim[2]))
        self.params['W5'] = np.random.normal(scale=weight_scale, size=(hidden_dim[2], num_classes))
        self.params['b5'] = np.zeros((num_classes))

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]
        W4, b4 = self.params["W4"], self.params["b4"]
        W5, b5 = self.params["W5"], self.params["b5"]

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        conv_param = {"stride": 1, "pad": 0}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}
        
        # **********************forward pass******************************

        out1, cache1 = conv_forward_naive(X, W1, b1, conv_param)
        out1_relu, cache1_relu = relu_forward(out1)
        out1_pool, cache1_pool = max_pool_forward_naive(out1_relu, pool_param)

        out2, cache2 = conv_forward_naive(out1_pool, W2, b2, conv_param)
        out2_relu, cache2_relu = relu_forward(out2)
        out2_pool, cache2_pool = max_pool_forward_naive(out2_relu, pool_param)

        out3, cache3 = affine_forward(out2_pool, W3, b3)
        out3_relu, cache3_relu = relu_forward(out3)

        out4, cache4 = affine_forward(out3_relu, W4, b4)
        out4_relu, cache4_relu = relu_forward(out4)

        scores, cache5 = affine_forward(out4_relu, W5, b5)

        if y is None:
            return scores

        # **********************backward pass******************************

        grads = {}

        loss, dx = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(W5 * W5)+np.sum(W4 * W4)+np.sum(W3 * W3)+np.sum(W2 * W2)+np.sum(W1 * W1))
        dx, grads["W5"], grads["b5"] = affine_backward(dx, cache5)
        dx = relu_backward(dx, cache4_relu)
        dx, grads["W4"], grads["b4"] = affine_backward(dx, cache4)
        dx = relu_backward(dx, cache3_relu)
        dx, grads["W3"], grads["b3"] = affine_backward(dx, cache3)
        dx = max_pool_backward_naive(dx, cache2_pool)
        dx = relu_backward(dx, cache2_relu)
        dx, grads["W2"], grads["b2"] = conv_backward_naive(dx, cache2)
        dx = max_pool_backward_naive(dx, cache1_pool)
        dx = relu_backward(dx, cache1_relu)
        _, grads["W1"], grads["b1"] = conv_backward_naive(dx, cache1)
        grads["W1"] += self.reg * self.params["W1"]
        grads["W2"] += self.reg * self.params["W2"]
        grads["W3"] += self.reg * self.params["W3"]
        grads["W4"] += self.reg * self.params["W4"]
        grads["W5"] += self.reg * self.params["W5"]

        return loss, grads
