import numpy as np

def conv_forward_naive(x, w, b, conv_param):
    """
    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    pad = conv_param['pad']
    stride = conv_param['stride']
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride
    out = np.zeros((N, F, H_out, W_out))
    for _x in range(H_out):
      for y in range(W_out):
        h_start = stride * _x
        w_start = stride * y
        in_permuted = padded[:, :, h_start:h_start+HH, w_start:w_start+WW].reshape(N, 1, C, HH, WW)
        w_permuted =  w.reshape(1, F, C, HH, WW)
        out[:, :, _x, y] = np.sum(in_permuted * w_permuted, axis=(-3, -2, -1))
        out[:, :, _x, y] += b
    cache = (x, w, b, conv_param)
    return out, cache

def conv_backward_naive(dout, cache):
    """
    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    input = cache[0]
    w = cache[1]
    b = cache[2]
    N, F, h_out, w_out = dout.shape
    _, _, H, W = input.shape
    _, C, HH, WW = w.shape
    pad = cache[3]['pad']
    stride = cache[3]['stride']
    padded = np.pad(input, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    din = np.zeros_like(padded)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    for x in range(h_out):
      for y in range(w_out):
        h_start = stride * x
        w_start = stride * y
        #STEP1: dx = dout * w
        y_spread = dout[:, :, x, y].reshape((N, 1, 1, 1, F))
        w_permute = w.transpose((1, 2, 3, 0)).reshape((1, C, HH, WW, F))
        din[:, :, h_start:h_start+HH, w_start:w_start+WW] += np.sum(y_spread * w_permute, axis=-1)
        #STEP2: dw = dout * x
        y_permute = dout[:, :, x, y].T.reshape((F, 1, 1, 1, N))
        x_transpose = padded[:, :, h_start:h_start+HH, w_start:w_start+WW].transpose(1, 2, 3, 0)
        dw += np.sum(y_permute * x_transpose, axis=-1)
        #STEP3: db = dy
        db += np.sum(dout[:, :, x, y], axis=0)
    dx = din[:, :, pad:H+pad, pad:W+pad]
    return dx, dw, db

def relu_forward(x):
    """
    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = np.maximum(0, x)
    cache = x
    return out, cache

def relu_backward(dout, cache):
    """
    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    x = cache
    x[x>0] = 1
    x[x<=0] = 0
    dx = dout * x
    return dx

def max_pool_forward_naive(x, pool_param):
    """
    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    N, C, H, W = x.shape
    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride
    out = np.zeros((N, C, H_out, W_out))
    for _x in range(H_out):
      for y in range(W_out):
        h_start = _x * stride
        w_start = y * stride
        out[:, :, _x, y] = np.max(x[:, :, h_start:h_start+pool_height, w_start:w_start+pool_width], axis=(2, 3))
    cache = (x, pool_param)
    return out, cache

def max_pool_backward_naive(dout, cache):
    """
    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    x = cache[0]
    pool_height = cache[1]['pool_height']
    pool_width = cache[1]['pool_width']
    stride = cache[1]['stride']
    N, C, H, W = x.shape
    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride
    dx = np.zeros_like(x)
    for _x in range(H_out):
      for y in range(W_out):
        h_start = _x * stride
        w_start = y * stride
        kernel = x[:, :, h_start:h_start+pool_height, w_start:w_start+pool_width].reshape(N, C, -1)
        idx_flattened = np.argmax(kernel, axis=-1)
        index = np.array(np.unravel_index(idx_flattened, (pool_height, pool_width)))
        for n in range(N):
          for c in range(C):
            dx[n][c][h_start+index[0][n][c]][w_start+index[1][n][c]] += dout[n][c][_x][y]
    return dx

def affine_forward(x, w, b):
    """
    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = x.reshape(x.shape[0], -1).dot(w) + b
    cache = (x, w, b)
    return out, cache

def affine_backward(dout, cache):
    """
    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx = dout.dot(w.T)
    dx = dx.reshape(x.shape)
    x_flat = x.reshape(x.shape[0], -1)
    dw = x_flat.T.dot(dout)
    db = np.sum(dout, axis = 0)
    return dx, dw, db

def softmax_loss(x, y):
    """
    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    inner = x
    ex_num = inner.shape[0]
    ex = np.exp(inner)
    sumx = np.sum(ex, axis = 1)
    loss = np.mean(np.log(sumx)-inner[range(ex_num), list(y)])
    dx = ex/sumx.reshape(ex_num, 1)
    dx[range(ex_num), list(y)] -= 1
    dx /= ex_num
    return loss, dx
