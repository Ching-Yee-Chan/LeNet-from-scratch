# coding=utf-8
import numpy as np
import struct
import os

# Load the MNIST data for this exercise
# mat_data contain the training and testing images or labels.
#   Each matrix has size [m,h,w] for images where:
#      m is the number of examples.
#      h*w is the number of pixels in each image.
#   or Each matrix has size [m,1] for labels contain the corresponding labels (0 to 9) where:
#      m is the number of examples.
def load_mnist(file_dir, is_images='True'):
    # Read binary data
    bin_file = open(file_dir, 'rb')
    bin_data = bin_file.read()
    bin_file.close()
    # Analysis file header
    if is_images:
        # Read images
        fmt_header = '>iiii'
        magic, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, 0)
    else:
        # Read labels
        fmt_header = '>ii'
        magic, num_images = struct.unpack_from(fmt_header, bin_data, 0)
        num_rows, num_cols = 1, 1
    data_size = num_images * num_rows * num_cols
    mat_data = struct.unpack_from('>' + str(data_size) + 'B', bin_data, struct.calcsize(fmt_header))
    if is_images:
        mat_data = np.reshape(mat_data, [num_images, num_rows, num_cols])
    else:
        mat_data = np.reshape(mat_data, [num_images])
    print('Load images from %s, number: %d, data shape: %s' % (file_dir, num_images, str(mat_data.shape)))
    return mat_data

# call the load_mnist function to get the images and labels of training set and testing set
def load_data(mnist_dir, train_data_dir, train_label_dir, test_data_dir, test_label_dir):
    print('Loading MNIST data from files...')
    train_images = load_mnist(os.path.join(mnist_dir, train_data_dir), True)
    train_labels = load_mnist(os.path.join(mnist_dir, train_label_dir), False)
    test_images = load_mnist(os.path.join(mnist_dir, test_data_dir), True)
    test_labels = load_mnist(os.path.join(mnist_dir, test_label_dir), False)
    return train_images, train_labels, test_images, test_labels

# 从训练集中划分出验证集，正则化
def get_mnist_data(num_validation=1000, subtract_mean=True):
    # Load the raw MNIST data
    mnist_dir = "mnist_data/"
    train_data_dir = "train-images.idx3-ubyte"
    train_label_dir = "train-labels.idx1-ubyte"
    test_data_dir = "t10k-images.idx3-ubyte"
    test_label_dir = "t10k-labels.idx1-ubyte"
    train_images, train_labels, test_images, test_labels = load_data(mnist_dir, train_data_dir, train_label_dir, test_data_dir, test_label_dir)
    
    # pad to 32, change type to float64, and reshape to [N, 1, H, W]
    train_images = np.pad(train_images, ((0, 0), (2, 2), (2, 2)))
    test_images = np.pad(test_images, ((0, 0), (2, 2), (2, 2)))

    N, H, W = train_images.shape
    train_images = train_images.astype(np.float32).reshape(-1, 1, H, W)
    test_images = test_images.astype(np.float32).reshape(-1, 1, H, W)

    # Subsample the validation set
    validation_images, validation_labels = [], []
    if num_validation!=0:
        validation_images = train_images[-num_validation:]
        validation_labels = train_labels[-num_validation:]
        train_images = train_images[:-num_validation]
        train_labels = train_labels[:-num_validation]
    
    # Normalize the data: subtract the mean image
    if subtract_mean:
        mean_image = np.mean(train_images, axis=0)
        train_images -= mean_image
        validation_images -= mean_image
        test_images -= mean_image

    # Package data into a dictionary
    return {
        "X_train": train_images,
        "y_train": train_labels,
        "X_val": validation_images,
        "y_val": validation_labels,
        "X_test": test_images,
        "y_test": test_labels,
    }