import time

import tensorflow as tf

from src.model import neural_net
from src import losses

class ConvNet1(neural_net.NeuralNet):
	"""A neural net for segmentation."""

	_WEIGHT_DECAY = 10e-5

	def __init__(self, save_dir, chkpt_save_interval):
		"""Create a conv-net-1 model.

		Args:
			save_dir (str): See neural_net.NeuralNet.
			chkpt_save_interval (int): See neural_net.NeuralNet.
		"""
		def relu(x, name):
			return tf.nn.relu(x, name=name)
		layers = [
		    neural_net.ConvolutionalLayer(5, 5, 1, "VALID", self._WEIGHT_DECAY,
		    	"layer1conv"),
		    neural_net.TransferLayer(relu, "layer1transfer"),
		    neural_net.PoolingLayer("MAX", 2, 1, "VALID", "layer1pooling"),
		    neural_net.ConvolutionalLayer(5, 4, 1, "VALID", self._WEIGHT_DECAY,
		    	"layer2conv"),
		    neural_net.TransferLayer(relu, "layer2transfer"),
		    neural_net.PoolingLayer("MAX", 2, 1, "VALID", "layer2pooling"),
		    neural_net.ConvolutionalLayer(5, 3, 1, "VALID", self._WEIGHT_DECAY,
		    	"layer3conv"),
		    neural_net.TransferLayer(relu, "layer3transfer"),
		    neural_net.PoolingLayer("MAX", 2, 1, "VALID", "layer3pooling"),
		    neural_net.FlatLayer("layer6flat"),
		    neural_net.FullLayer(100, self._WEIGHT_DECAY, "layer6full"),
		    neural_net.TransferLayer(relu, "layer6transfer"),
		    neural_net.FullLayer(100, self._WEIGHT_DECAY, "layer7full"),
		    neural_net.TransferLayer(relu, "layer7transfer"),
		    neural_net.FullLayer(1, self._WEIGHT_DECAY, "layer9full")
		]
		loss_fn = losses.mse_loss
		def optimizer_factory(global_step):
			learning_rate = tf.train.exponential_decay(0.01,
				global_step, 1, 0.95, staircase=False)
			return tf.train.MomentumOptimizer(learning_rate, 0.2)
		super().__init__(save_dir, chkpt_save_interval, loss_fn,
			optimizer_factory, layers)

	def _get_batch_size(self):
		return 256
