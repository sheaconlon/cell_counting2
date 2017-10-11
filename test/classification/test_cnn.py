import unittest

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from src.classification import cnn
from src.classification import metrics

class CNNTestCase(unittest.TestCase):
    """Tests for cnn.py"""

    def test_neural_net_mnist_all_layer_types(self):
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        layers = [
            cnn.ConvolutionalLayer(1, 5, 1, "VALID", "bla1"),
            cnn.PoolingLayer("AVG", 3, 1, "VALID", "bla2"),
            cnn.TransferLayer(lambda x, name: tf.sigmoid(x, name=name), "bla3"),
            cnn.FlatLayer("bla4"),
            cnn.FullLayer(10, "bla6")
        ]
        net = cnn.NeuralNet(
            layers,
            lambda labels, predictions: tf.losses.softmax_cross_entropy(
                labels,
                predictions,
                weights=1,
                label_smoothing=1,
                scope=None,
                loss_collection=tf.GraphKeys.LOSSES,
                reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
            ),
            {
                "accuracy": metrics.accuracy
            },
            lambda: tf.train.RMSPropOptimizer(0.5, decay=0.9, momentum=0.01,
                epsilon=1e-10, use_locking=False, centered=False, name="bla7"),
            "test_neural_net_mnist_all_layer_types",
            "bla8"  
        )
        tf_train_data_fn = tf.estimator.inputs.numpy_input_fn(
            {"images":mnist.train.images},
            y=mnist.train.labels,
            batch_size=128,
            num_epochs=None,
            shuffle=True,
            queue_capacity=1000,
            num_threads=8
        )
        def train_data_fn():
            x, y = tf_train_data_fn()
            return tf.expand_dims(tf.reshape(tf.expand_dims(x["images"], axis=2), shape=(128, 28, 28)), axis=3), y
        net.train_duration_(train_data_fn, 30)
        tf_test_data_fn = tf.estimator.inputs.numpy_input_fn(
            {"images":mnist.test.images},
            y=mnist.test.labels,
            batch_size=100,
            num_epochs=10,
            shuffle=False,
            queue_capacity=1000,
            num_threads=1
        )
        def test_data_fn():
            x, y = tf_test_data_fn()
            return tf.expand_dims(tf.reshape(tf.expand_dims(x["images"], axis=2), shape=(100, 28, 28)), axis=3), y
        metric_dict = net.evaluate_(test_data_fn)
        print(metric_dict["accuracy"])
        self.assertGreaterEqual(metric_dict["accuracy"], 0.8)

    def test_neural_net_mnist_basic(self):
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        layers = [
            cnn.FlatLayer("bla4"),
            cnn.FullLayer(10, "bla6")
        ]
        net = cnn.NeuralNet(
            layers,
            lambda labels, predictions: tf.losses.softmax_cross_entropy(
                labels,
                predictions,
                weights=1,
                label_smoothing=0,
                scope=None,
                loss_collection=tf.GraphKeys.LOSSES,
                reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
            ),
            {
                "accuracy": metrics.accuracy
            },
            lambda: tf.train.GradientDescentOptimizer(0.5),
            "test_neural_net_mnist_basic",
            "bla8"  
        )
        tf_train_data_fn = tf.estimator.inputs.numpy_input_fn(
            {"images":mnist.train.images},
            y=mnist.train.labels,
            batch_size=100,
            num_epochs=5,
            shuffle=True,
            queue_capacity=1,
            num_threads=1
        )
        def train_data_fn():
            x, y = tf_train_data_fn()
            return tf.expand_dims(tf.reshape(tf.expand_dims(x["images"], axis=2), shape=(100, 28, 28)), axis=3), y
        net.train_(train_data_fn)
        tf_test_data_fn = tf.estimator.inputs.numpy_input_fn(
            {"images":mnist.test.images},
            y=mnist.test.labels,
            batch_size=100,
            num_epochs=10,
            shuffle=False,
            queue_capacity=1000,
            num_threads=1
        )
        def test_data_fn():
            x, y = tf_test_data_fn()
            return tf.expand_dims(tf.reshape(tf.expand_dims(x["images"], axis=2), shape=(100, 28, 28)), axis=3), y
        metric_dict = net.evaluate_(test_data_fn)
        print(metric_dict["accuracy"])
        self.assertGreaterEqual(metric_dict["accuracy"], 0.9)