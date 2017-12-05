import time

import tensorflow as tf

from src.model import neural_net

class ConvNet1(neural_net.NeuralNet):
	"""A neural net for segment counting."""

	def __init__(self, save_dir, chkpt_save_interval):
		"""Create a conv-net-1 model.

		Args:
			save_dir (str): See neural_net.NeuralNet.
			chkpt_save_interval (int): See neural_net.NeuralNet.
		"""
		def relu(x, name):
			return tf.nn.relu(x, name=name)
		weight_init = tf.contrib.keras.initializers.lecun_uniform()
		bias_init = tf.ones_initializer()
		weight_reg = tf.contrib.layers.l2_regularizer(0.0005)
		bias_reg = tf.contrib.layers.l2_regularizer(0.0005)
		layers = [
		    neural_net.ConvolutionalLayer(20, 5, 1, "VALID",
		    	weight_init, bias_init, weight_reg, bias_reg, "1-CONV"),
		    neural_net.TransferLayer(relu, "1-TRANSFER"),
		    neural_net.LocalResponseNormalizationLayer(1, 1, 1e-5, 0.5,
		    	"1-NORM"),
		    neural_net.PoolingLayer("MAX", 2, 2, "VALID", "1-POOL"),

		    neural_net.ConvolutionalLayer(50, 5, 1, "VALID",
		    	weight_init, bias_init, weight_reg, bias_reg, "2-CONV"),
		    neural_net.TransferLayer(relu, "2-TRANSFER"),
		    neural_net.LocalResponseNormalizationLayer(2, 1, 1e-5, 0.5,
		    	"2-NORM"),
		    neural_net.PoolingLayer("MAX", 2, 2, "VALID", "2-POOL"),

		    neural_net.ConvolutionalLayer(100, 4, 1, "VALID",
		    	weight_init, bias_init, weight_reg, bias_reg, "3-CONV"),
		    neural_net.TransferLayer(relu, "3-TRANSFER"),
		    neural_net.LocalResponseNormalizationLayer(5, 1, 1e-5, 0.5,
		    	"3-NORM"),
		    neural_net.PoolingLayer("MAX", 2, 2, "VALID", "3-POOL"),

		    neural_net.ConvolutionalLayer(200, 4, 1, "VALID",
		    	weight_init, bias_init, weight_reg, bias_reg, "4-CONV"),
		    neural_net.TransferLayer(relu, "4-TRANSFER"),
		    neural_net.LocalResponseNormalizationLayer(7, 1, 1e-5, 0.5,
		    	"4-NORM"),
		    neural_net.PoolingLayer("MAX", 2, 2, "VALID", "4-POOL"),

		    neural_net.FlatLayer("5-FLAT"),
		    neural_net.DropoutLayer(1 - 0.75, None, time.time(), "5-DROP"),
		    neural_net.FullLayer(500, weight_init, bias_init, weight_reg,
		    	bias_reg, "5-FULL"),
		    neural_net.TransferLayer(relu, "5-TRANSFER"),

		    neural_net.DropoutLayer(1 - 0.75, None, time.time(), "6-DROP"),
		    neural_net.FullLayer(7, weight_init, bias_init, weight_reg,
		    	bias_reg, "6-FULL")
		]
		def loss_fn(labels, predictions):
			predictions = tf.add(predictions, tf.constant(1e-4))
			loss = tf.losses.softmax_cross_entropy(labels, predictions)
			return loss
		def optimizer_factory(global_step):
			learning_rate = 0.01
			learning_rate = tf.train.exponential_decay(learning_rate,
				global_step, 1, 1 - 0.0001)
			learning_rate = tf.train.piecewise_constant(global_step, [20_000,
				40_000], [learning_rate, tf.multiply(learning_rate, 0.5),
				tf.multiply(learning_rate, 0.25)])
			return tf.train.MomentumOptimizer(learning_rate, 0.9)
		super().__init__(save_dir, chkpt_save_interval, loss_fn,
			optimizer_factory, layers)

	def _get_batch_size(self):
		return 64
