import time

import tensorflow as tf

from src.model import neural_net
from src import losses

class ConvNet2(neural_net.NeuralNet):
	"""A convolutional neural net for segment counting.

	Based off of ConvNet1.

	Uses class-difference loss with these parameters:
		class_fn: lambda cls: 0 if cls == 7 else cls + 1
		exp: 1.5

	Uses higher learning rate and learning rate decay.
		learning_rate: 0.1
		learning_rate_decay: 1 - 0.001

	Uses lower momemtum.
		momentum: 0.5

	Uses larger batch size.
		batch_size: 256

	Removed fourth layer.

	Reduced units in first fully-connected layer.
		units: 200

	Switched activation function from ReLU to leaky ReLU.
		alpha: 0.0001

		\smaller weight decay 10e-5
	"""

	BATCH_SIZE = 256

	WEIGHT_DECAY = 10e-5
	LEARNING_RATE = 0.03
	MOMENTUM = 0.5
	LEARNING_RATE_DECAY = 1 - 0.01

	LAYER_1_FILTERS = 20
	LAYER_1_WINDOW = 5
	LAYER_1_STRIDE = 1
	LAYER_1_CONV_PADDING = "VALID"
	LAYER_1_POOLING_TYPE = "MAX"
	LAYER_1_POOLING_WINDOW = 2
	LAYER_1_POOLING_PADDING = "VALID"

	LAYER_2_FILTERS = 50
	LAYER_2_WINDOW = 5
	LAYER_2_STRIDE = 1
	LAYER_2_CONV_PADDING = "VALID"
	LAYER_2_POOLING_TYPE = "MAX"
	LAYER_2_POOLING_WINDOW = 2
	LAYER_2_POOLING_PADDING = "VALID"

	LAYER_3_FILTERS = 100
	LAYER_3_WINDOW = 4
	LAYER_3_STRIDE = 1
	LAYER_3_CONV_PADDING = "VALID"
	LAYER_3_POOLING_TYPE = "MAX"
	LAYER_3_POOLING_WINDOW = 2
	LAYER_3_POOLING_PADDING = "VALID"

	FULLY_CONNECTED_DROPOUT_RATE = 0.1

	LAYER_5_UNITS = 200

	LAYER_6_UNITS = 7

	def __init__(self, save_dir, chkpt_save_interval):
		"""Create a conv-net-1 model.

		Args:
			save_dir (str): See neural_net.NeuralNet.
			chkpt_save_interval (int): See neural_net.NeuralNet.
		"""
		def relu(x, name):
			return tf.nn.leaky_relu(x, alpha=0.00005, name=name)
		layers = [
		    neural_net.ConvolutionalLayer(self.LAYER_1_FILTERS, self.LAYER_1_WINDOW,
		    	self.LAYER_1_STRIDE, self.LAYER_1_CONV_PADDING,
		    	self.WEIGHT_DECAY, "layer1conv"),
		    neural_net.LocalResponseNormalizationLayer(5, 1, 1, 0.5, "layer1lrn"),
		    neural_net.TransferLayer(relu, "layer1transfer"),
		    neural_net.PoolingLayer(self.LAYER_1_POOLING_TYPE,
		    	self.LAYER_1_POOLING_WINDOW, self.LAYER_1_POOLING_WINDOW,
		    	self.LAYER_1_POOLING_PADDING, "layer1pooling"),
		    neural_net.ConvolutionalLayer(self.LAYER_2_FILTERS, self.LAYER_2_WINDOW,
		    	self.LAYER_2_STRIDE, self.LAYER_2_CONV_PADDING,
		    	self.WEIGHT_DECAY, "layer2conv"),
		    neural_net.LocalResponseNormalizationLayer(5, 1, 1, 0.5, "layer2lrn"),
		    neural_net.TransferLayer(relu, "layer2transfer"),
		    neural_net.PoolingLayer(self.LAYER_2_POOLING_TYPE,
		    	self.LAYER_2_POOLING_WINDOW, self.LAYER_2_POOLING_WINDOW,
		    	self.LAYER_2_POOLING_PADDING, "layer2pooling"),
		    neural_net.ConvolutionalLayer(self.LAYER_3_FILTERS, self.LAYER_3_WINDOW,
		    	self.LAYER_3_STRIDE, self.LAYER_3_CONV_PADDING,
		    	self.WEIGHT_DECAY, "layer3conv"),
		    neural_net.LocalResponseNormalizationLayer(5, 1, 1, 0.5, "layer3lrn"),
		    neural_net.TransferLayer(relu, "layer3transfer"),
		    neural_net.PoolingLayer(self.LAYER_3_POOLING_TYPE,
		    	self.LAYER_3_POOLING_WINDOW, self.LAYER_3_POOLING_WINDOW,
		    	self.LAYER_3_POOLING_PADDING, "layer3pooling"),
		    neural_net.DropoutLayer(self.FULLY_CONNECTED_DROPOUT_RATE, None,
		    	time.time(), "layer4dropout"),
		    neural_net.FlatLayer("layer5flat"),
		    neural_net.FullLayer(self.LAYER_5_UNITS, self.WEIGHT_DECAY, "layer5full"),
		    neural_net.TransferLayer(relu, "layer5transfer"),
		    neural_net.DropoutLayer(self.FULLY_CONNECTED_DROPOUT_RATE, None,
		    	time.time(), "layer5dropout"),
		    neural_net.FullLayer(self.LAYER_6_UNITS, self.WEIGHT_DECAY, "layer6full")
		]
		def loss_fn(labels, predictions):
			return tf.losses.softmax_cross_entropy(labels, predictions,
				reduction=tf.losses.Reduction.SUM)
		def optimizer_factory(global_step):
			learning_rate = tf.train.exponential_decay(self.LEARNING_RATE,
				global_step, 1, self.LEARNING_RATE_DECAY, staircase=False)
			return tf.train.MomentumOptimizer(learning_rate, self.MOMENTUM)
		super().__init__(save_dir, chkpt_save_interval, loss_fn,
			optimizer_factory, layers)

	def _get_batch_size(self):
		return self.BATCH_SIZE
