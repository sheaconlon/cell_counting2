import math

import tensorflow as tf
import numpy as np

from src import neural_net
from src import losses

class ConvNet3(neural_net.NeuralNet):
	"""A convolutional neural net for the segmentation of cell colonies in
		images of plates. Based off of the model for the segmentation
		of bacterial cell cytoplasm in microscopy images in Valen et al. Images
		must have shape (61, 61, n_channels)."""

	_CLASS_COUNT_JITTER = 0.1

	def __init__(self, save_dir, chkpt_save_interval, num_samples):
		"""Create a conv-net-1 model.

		Args:
			save_dir (str): See neural_net.NeuralNet.
			chkpt_save_interval (int): See neural_net.NeuralNet.
			num_samples (int): The number of samples in the training dataset.
		"""
		WEIGHT_DECAY = 1e-5
		def conv(n, size, stride, name):
			return neural_net.ConvolutionalLayer(n, size, stride, "VALID",
		    	weight_init=tf.keras.initializers.he_normal(),
		    	bias_init=tf.keras.initializers.he_normal(),
		    	weight_reg=tf.contrib.layers.l2_regularizer(WEIGHT_DECAY),
		    	name=name)
		def batch(name):
			return neural_net.BatchNormalizationLayer(3, name)
		def relu(name):
			return neural_net.TransferLayer(
				lambda x, name: tf.nn.relu(x, name=name), name)
		def pool(name):
			return neural_net.PoolingLayer("MAX", 2, 2, "VALID", name)
		def full(n, name):
			return neural_net.FullLayer(n, tf.keras.initializers.he_normal(),
		    	tf.keras.initializers.he_normal(),
		    	weight_reg=tf.contrib.layers.l2_regularizer(WEIGHT_DECAY),
		    	name=name)
		layers = (
		    conv(64, 3, 1, "l1-conv"),
		    batch("l1-batch"),
		    relu("l1-relu"),

		    conv(64, 4, 1, "l2-conv"),
		    batch("l2-batch"),
		    relu("l2-relu"),
		    pool("l2-pool"),

		    conv(64, 3, 1, "l3-conv"),
		    batch("l3-batch"),
		    relu("l3-relu"),

		    conv(64, 3, 1, "l4-conv"),
		    batch("l4-batch"),
		    relu("l4-relu"),
		    pool("l4-pool"),

		    conv(64, 3, 1, "l5-conv"),
		    batch("l5-batch"),
		    relu("l5-relu"),

		    conv(64, 3, 1, "l6-conv"),
		    batch("l6-batch"),
		    relu("l6-relu"),
		    pool("l6-pool"),

		    conv(200, 4, 1, "l7-conv"),
		    batch("l7-batch"),
		    relu("l7-relu"),

		    neural_net.FlatLayer("flat"),

		    full(200, "l9-full"),
		    neural_net.BatchNormalizationLayer(1, "l8-batch"),
		    relu("l8-relu"),

		   	full(3, "l10-full")
		)

		def loss_fn(actual, predicted):
			assert 0 < self._CLASS_COUNT_JITTER < 1

			class_counts = tf.reduce_sum(actual, axis=0)
			class_count_jitter = tf.constant(self._CLASS_COUNT_JITTER)
			# Make the classes with count 0 have count class_count_jitter. The
			# following line will affect only those classes with count 0 so long
			# as self._CLASS_COUNT_JITTER < 1.
			class_counts = tf.maximum(class_counts, class_count_jitter)
			total_count = tf.cast(tf.shape(actual)[0], tf.float32)
			class_props = tf.divide(class_counts, total_count)
			# This will not produce any NaNs because all counts are greater than
			# 0. It will not matter what weight is used for classes with count
			# 0 because the weight will be multiplied by 0 in each row of
			# actual.
			class_weights = tf.reciprocal(class_props)
			class_weights = tf.expand_dims(class_weights, 1)
			weights = tf.matmul(actual, class_weights)
			weights = weights[:, 0]
			loss = tf.losses.softmax_cross_entropy(actual, predicted,
				weights=weights)
			loss = tf.Print(loss, data=[actual, weights], summarize=100)
			return loss

		def optimizer_factory(global_step):
			learning_rate = tf.train.exponential_decay(0.01, global_step, 1,
				1 - 1e-6, staircase=True)
			learning_rate = tf.train.exponential_decay(learning_rate,
				global_step, num_samples, 0.95, staircase=True)
			return tf.train.MomentumOptimizer(learning_rate, 0.9,
				use_nesterov=True)
		super().__init__(save_dir, chkpt_save_interval, loss_fn,
			optimizer_factory, layers)

	def get_batch_size(self):
		return 256
