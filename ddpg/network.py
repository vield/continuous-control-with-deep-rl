"""Helper methods to create fully-connected network layers."""
import tensorflow as tf


def create_bias_shaped_variables(nodes_per_layer,
                                 mean=None, stddev=None,
                                 name_prefix="Biases",
                                 trainable=True):
    """Does what it says on the tin.

    Parameters
    ----------
    nodes_per_layer : list of integers
        E.g. [784, 100, 100, 10] means that there are 784 features (pixels)
        coming in, two hidden layers with 100 nodes each, and the output
        vector has length 10.
        The bias variables will be created to match this structure.
    mean : float
    stddev : float
        If set to a truthy value, the bias-shaped variable will be initialized
        from a truncated normal distribution with the given mean (default 0.0)
        and the stddev. Otherwise, it will be initialized to the mean.
    name_prefix : string
        Used to name the tensors.
    trainable : bool
        Passed into the Variable constructor to make the tensor trainable
        by default, or not if trainable=False.
    """
    biases = []

    if mean is None:
        mean = 0.0

    for layer_idx in range(1, len(nodes_per_layer)):
        num_out = nodes_per_layer[layer_idx]
        shape = [num_out]

        if stddev:
            initial = tf.truncated_normal(shape=shape, stddev=stddev, mean=mean)
        else:
            initial = tf.constant(mean, shape=shape)

        b = tf.Variable(
            initial,
            name=name_prefix + str(layer_idx),
            trainable=trainable
        )
        biases.append(b)

    return biases


def create_weight_shaped_variables(nodes_per_layer,
                                   mean=None, stddev=None,
                                   name_prefix="Weights",
                                   trainable=True):
    """Same as bias-shaped variables except this is for weights. See other docstring."""
    weights = []

    if mean is None:
        mean = 0.0

    for layer_idx in range(1, len(nodes_per_layer)):
        num_in = nodes_per_layer[layer_idx-1]
        num_out = nodes_per_layer[layer_idx]
        shape = [num_in, num_out]

        if stddev:
            initial = tf.truncated_normal(shape=shape, stddev=stddev, mean=mean)
        else:
            initial = tf.constant(mean, shape=shape)

        W = tf.Variable(
            initial,
            name=name_prefix + str(layer_idx),
            trainable=trainable
        )
        weights.append(W)

    return weights


def create_fully_connected_architecture(inputs, biases, weights):
    """Creates fully connected layers out of the inputs, biases and weights.

    All layers except the last go through ReLU activation.

    Nodes per layer:
        [n_1, n_2, ..., n_n]

    :param inputs:
        tf.placeholder for your input variables. Should have shape
        [X, n_1] where X may be None or any size you've chosen it to be.
    :param biases:
        One-dimensional add-on values. Should have dimensions [(n_2,), ..., (n_n,)].
    :param weights:
        Weights to convert between layers through tf.matmul.
        Should have dimensions [(n_1, n_2), (n_2, n_3), ..., (n_n-1, n_n)].
    :return:
        Unscaled, unactivated outputs. Dimensions are [X, n_n].
    """
    # Last layer is a no-op
    activation_functions = [tf.nn.relu] * (len(biases) - 1) + [tf.identity]

    prev = inputs
    for layer_idx in range(len(biases)):
        b = biases[layer_idx]
        W = weights[layer_idx]
        act_func = activation_functions[layer_idx]

        y = act_func(tf.matmul(prev, W) + b)
        prev = y

    return prev
