import time

import tensorflow as tf

from src.model import neural_net

class ConvNet1(neural_net.NeuralNet):
	"""A neural net for segment counting."""

	BATCH_SIZE = 64

	WEIGHT_DECAY = 0.0005
	LEARNING_RATE = 0.01
	MOMENTUM = 0.9
	LEARNING_RATE_DECAY = 1 - 0.0001

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

	LAYER_4_FILTERS = 200
	LAYER_4_WINDOW = 4
	LAYER_4_STRIDE = 1
	LAYER_4_CONV_PADDING = "VALID"
	LAYER_4_POOLING_TYPE = "MAX"
	LAYER_4_POOLING_WINDOW = 2
	LAYER_4_POOLING_PADDING = "VALID"
	LAYER_4_DROPOUT_RATE = 0.1

	LAYER_5_UNITS = 500
	LAYER_5_DROPOUT_RATE = 0.1

	LAYER_6_UNITS = 7

	LABEL_SMOOTHING = 0.1

	def __init__(self, save_dir, chkpt_save_interval):
		"""Create a conv-net-1 model.

		Args:
			save_dir (str): See neural_net.NeuralNet.
			chkpt_save_interval (int): See neural_net.NeuralNet.
		"""
		def relu(x, name):
			return tf.nn.relu(x, name=name)
		layers = [
		    neural_net.ConvolutionalLayer(self.LAYER_1_FILTERS, self.LAYER_1_WINDOW,
		    	self.LAYER_1_STRIDE, self.LAYER_1_CONV_PADDING,
		    	self.WEIGHT_DECAY, "layer1conv"),
		    neural_net.TransferLayer(relu, "layer1transfer"),
		    neural_net.LocalResponseNormalizationLayer(5, 1, 1, 0.5, "layer1lrn"),
		    neural_net.PoolingLayer(self.LAYER_1_POOLING_TYPE,
		    	self.LAYER_1_POOLING_WINDOW, self.LAYER_1_POOLING_WINDOW,
		    	self.LAYER_1_POOLING_PADDING, "layer1pooling"),
		    neural_net.ConvolutionalLayer(self.LAYER_2_FILTERS, self.LAYER_2_WINDOW,
		    	self.LAYER_2_STRIDE, self.LAYER_2_CONV_PADDING,
		    	self.WEIGHT_DECAY, "layer2conv"),
		    neural_net.TransferLayer(relu, "layer2transfer"),
		    neural_net.LocalResponseNormalizationLayer(5, 1, 1, 0.5, "layer2lrn"),
		    neural_net.PoolingLayer(self.LAYER_2_POOLING_TYPE,
		    	self.LAYER_2_POOLING_WINDOW, self.LAYER_2_POOLING_WINDOW,
		    	self.LAYER_2_POOLING_PADDING, "layer2pooling"),
		    neural_net.ConvolutionalLayer(self.LAYER_3_FILTERS, self.LAYER_3_WINDOW,
		    	self.LAYER_3_STRIDE, self.LAYER_3_CONV_PADDING,
		    	self.WEIGHT_DECAY, "layer3conv"),
		    neural_net.TransferLayer(relu, "layer3transfer"),
		    neural_net.LocalResponseNormalizationLayer(5, 1, 1, 0.5, "layer3lrn"),
		    neural_net.PoolingLayer(self.LAYER_3_POOLING_TYPE,
		    	self.LAYER_3_POOLING_WINDOW, self.LAYER_3_POOLING_WINDOW,
		    	self.LAYER_3_POOLING_PADDING, "layer3pooling"),
		    neural_net.ConvolutionalLayer(self.LAYER_4_FILTERS, self.LAYER_4_WINDOW,
		    	self.LAYER_4_STRIDE, self.LAYER_4_CONV_PADDING,
		    	self.WEIGHT_DECAY, "layer4conv"),
		    neural_net.TransferLayer(relu, "layer4transfer"),
		    neural_net.LocalResponseNormalizationLayer(5, 1, 1, 0.5, "layer4lrn"),
		    neural_net.PoolingLayer(self.LAYER_4_POOLING_TYPE,
		    	self.LAYER_4_POOLING_WINDOW, self.LAYER_4_POOLING_WINDOW,
		    	self.LAYER_4_POOLING_PADDING, "layer4pooling"),
		    neural_net.DropoutLayer(self.LAYER_4_DROPOUT_RATE, None, time.time(),
		    	"layer4dropout"),
		    neural_net.FlatLayer("layer5flat"),
		    neural_net.FullLayer(self.LAYER_5_UNITS, self.WEIGHT_DECAY, "layer5full"),
		    neural_net.TransferLayer(relu, "layer5transfer"),
		    neural_net.DropoutLayer(self.LAYER_5_DROPOUT_RATE, None, time.time(),
		    	"layer5dropout"),
		    neural_net.FullLayer(self.LAYER_6_UNITS, self.WEIGHT_DECAY, "layer6full")
		]
		def loss_fn(labels, predictions):
			return tf.losses.softmax_cross_entropy(labels, predictions,
				label_smoothing=self.LABEL_SMOOTHING,
				reduction=tf.losses.Reduction.SUM)
		def optimizer_factory(global_step):
			learning_rate = tf.train.exponential_decay(self.LEARNING_RATE,
				global_step, 1, self.LEARNING_RATE_DECAY, staircase=False)
			return tf.train.MomentumOptimizer(learning_rate, self.MOMENTUM)
		super().__init__(save_dir, chkpt_save_interval, loss_fn,
			optimizer_factory, layers)

	def _get_batch_size(self):
		return self.BATCH_SIZE
