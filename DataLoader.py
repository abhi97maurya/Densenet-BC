import os
import pickle
import numpy as np

"""This script implements the functions for reading data.
"""

def load_data(data_dir):
    """Load the CIFAR-10 dataset.

    Args:
        data_dir: A string. The directory where data batches
            are stored.

    Returns:
        x_train: An numpy array of shape [50000, 3072].
            (dtype=np.float32)
        y_train: An numpy array of shape [50000,].
            (dtype=np.int32)
        x_test: An numpy array of shape [10000, 3072].
            (dtype=np.float32)
        y_test: An numpy array of shape [10000,].
            (dtype=np.int32)
    """

    ### YOUR CODE HERE
    files = os.listdir(data_dir)
    x_train = np.array([[]]).reshape(0,3072)
    y_train = np.array([])
    for fn in files:
        if fn.endswith("html") or fn.endswith("meta") or fn.endswith("pdf"):
            continue
        with open(os.path.join(data_dir,fn), 'rb') as fo:
            ds = pickle.load(fo, encoding='bytes')
            xtemp = np.array(ds[b'data']) 
            ytemp = np.array(ds[b'labels'])

        if fn.startswith("test"):
            x_test = xtemp
            y_test = ytemp

        if fn.startswith("data"):
            x_train = np.concatenate((xtemp,x_train), axis=0)
            y_train = np.concatenate((ytemp,y_train), axis=0)
    return x_train, y_train, x_test, y_test
    ### END CODE HERE

def load_testing_images(data_dir):
    """Load the images in private testing dataset.

    Args:
        data_dir: A string. The directory where the testing images
        are stored.

    Returns:
        x_test: An numpy array of shape [N, 3072].
            (dtype=np.float32)
    """

    ### YOUR CODE HERE
    files = os.listdir(data_dir)
    x_test = np.array([[]]).reshape(0,3072)
    for fn in files:
        if fn.endswith("html") or fn.endswith("meta") or fn.endswith("pdf"):
            continue
        xtemp = np.load(os.path.join(data_dir,fn))

        if fn.startswith("test"):
            x_test = xtemp

        if fn.startswith("data") or fn.startswith("private"):
            x_test = np.concatenate((xtemp,x_test), axis=0)
    return x_test
    ### END CODE HERE

    return x_test


def train_valid_split(x_train, y_train, train_ratio=0.8):
    """Split the original training data into a new training dataset
    and a validation dataset.

    Args:
        x_train: An array of shape [50000, 3072].
        y_train: An array of shape [50000,].
        train_ratio: A float number between 0 and 1.

    Returns:
        x_train_new: An array of shape [split_index, 3072].
        y_train_new: An array of shape [split_index,].
        x_valid: An array of shape [50000-split_index, 3072].
        y_valid: An array of shape [50000-split_index,].
    """
    
    ### YOUR CODE HERE
    split_index = int(train_ratio * x_train.shape[0])
    x_train_new = x_train[:split_index]
    y_train_new = y_train[:split_index]
    x_valid = x_train[split_index:]
    y_valid = y_train[split_index:]
    ### END CODE HERE

    return x_train_new, y_train_new, x_valid, y_valid

