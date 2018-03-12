import time

import tensorflow as tf
import numpy as np

from src import model

tf.logging.set_verbosity(tf.logging.ERROR)

class NeuralNet(model.BaseModel):
	def __init__(self, save_dir, chkpt_save_interval, loss_fn,
			optimizer_factory, layers):
		"""Create a neural net.

		Args:
			save_dir (str): See model.BaseModel.
			chkpt_save_interval (int): See model.BaseModel.
			loss_fn (func(tf.Tensor, tf.Tensor) -> tf.Tensor): See
				model.BaseModel.
			optimizer_factory (func(tf.Tensor) -> tf.train.Optimizer): See
				model.BaseModel.
			layers (list of neural_net.BaseLayer): The layers of the neural net.
		"""
		super().__init__(save_dir, chkpt_save_interval, loss_fn,
			optimizer_factory)
		self._layers = layers

	def _tensor_predict(self, inputs):
		with tf.Session().as_default():
			result = inputs
			# result = tf.Print(result, [result, tf.reduce_mean(result)], summarize=20)
			for layer in self._layers:
				result = layer.output(result)
				# result = tf.Print(result, [result, tf.reduce_mean(result)], summarize=20)
			return result

class BaseLayer(object):
	TYPES = set(["CONV", "POOL", "TRAN", "LRN", "FLAT", "FULL", "DROP"])
	DATA_FORMAT = "channels_last"

	"""
	A layer of a neural network.
	"""
	def __init__(self, layer_type):
		"""
		Create a layer.

		Args:
			layer_type (str): One of `"CONV"`, `"POOL"`, `"TRAN"`, `"FLAT"`, or `"FULL"`.
		"""
		assert layer_type in BaseLayer.TYPES
		self._type = layer_type

	def type(self):
		"""
		Get the type of this layer.

		Returns:
			A string giving the type of this layer. One of `{"CONV", "POOL", "TRAN", "FLAT", "FULL"}`.
		"""
		return self._type


	def output(self, previous):
		"""
		Get the output of this layer.

		Args:
			previous (tf.Tensor): The output of the layer before this one.

		Returns:
			A `tf.Tensor` representing the output of this layer.
		"""
		#assert type(previous) == tf.Tensor

class ConvolutionalLayer(BaseLayer):
	"""
	A convolutional layer of a neural network.
	"""

	TYPE = "CONV"
	PADDING_METHODS = set(["VALID", "SAME"])
	DILATION_RATE = 1
	ACTIVATION = None
	USE_BIAS = True
	ACTIVITY_REGULARIZER = None
	TRAINABLE = True
	REUSE = False

	def __init__(self, n_filters, window_side, window_stride, padding_method,
		weight_init, bias_init=None, weight_reg=None,
		bias_reg=None, name=None):
		"""
		Create a convolutional layer.

		Args:
			n_filters (int): Number of filters.
			window_side_length (int): The side length of the sliding window for the filters, in pixels.
			stride_length (int): The stride length of the sliding window for the filters, in pixels.
			padding_method (str): The name of the padding method to use.
			weight_init (like return value of
				tf.contrib.keras.initializers.Initializer): An initializer for
				the weights.
			bias_init (like return value of
				tf.contrib.keras.initializers.Initializer): An initializer for
				the biases. If omitted or None, then no biases are used.
			weight_reg (like return value of tf.contrib.layers.l2_regularizer):
				A regularizer for the weights. If omitted or None, then no
				regularization is applied to the weights.
			bias_reg (like return value of tf.contrib.layers.l2_regularizer):
				A regularizer for the biases. If omitted or None, then no
				regulatization is applied to the biases.
			name (str): The name to use for any `tf.Tensor`s created.
		"""
		super().__init__(ConvolutionalLayer.TYPE)
		assert isinstance(n_filters, int)
		assert n_filters >= 1
		assert isinstance(window_side, int)
		assert window_side >= 1
		assert isinstance(window_stride, int)
		assert window_stride >= 1
		assert padding_method in ConvolutionalLayer.PADDING_METHODS
		assert isinstance(weight_init,
			tf.contrib.keras.initializers.Initializer)
		assert bias_init is None or \
			isinstance(bias_init, tf.contrib.keras.initializers.Initializer)
		self._n_filters = n_filters
		self._window_side = window_side
		self._window_stride = window_stride
		self._padding_method = padding_method
		self._weight_init = weight_init
		self._bias_init = bias_init
		self._weight_reg = weight_reg
		self._bias_reg = bias_reg
		self._name = name

	def output(self, previous):
		"""
		Get the output of this layer.

		Args:
			previous (tf.Tensor): The output of the layer before this one. Must have shape `[n_batches, height, width, channels]`.

		Returns:
			A `tf.Tensor` representing the output of this layer.
		"""
		super().output(previous)
		return tf.layers.conv2d(
			previous,
			self._n_filters,
			self._window_side,
			strides=self._window_stride,
			padding=self._padding_method,
			data_format=BaseLayer.DATA_FORMAT,
			dilation_rate=ConvolutionalLayer.DILATION_RATE,
			activation=ConvolutionalLayer.ACTIVATION,
			use_bias=ConvolutionalLayer.USE_BIAS,
			kernel_initializer=self._weight_init,
			bias_initializer=self._bias_init,
			kernel_regularizer=self._weight_reg,
			bias_regularizer=self._bias_reg,
			activity_regularizer=ConvolutionalLayer.ACTIVITY_REGULARIZER,
			trainable=ConvolutionalLayer.TRAINABLE,
			name=self._name,
			reuse=ConvolutionalLayer.REUSE
		)

class LocalResponseNormalizationLayer(BaseLayer):
	"""
	A local response normalization layer of a neural network.
	"""

	TYPE = "LRN"

	def __init__(self, depth_radius, bias, alpha, beta, name):
		"""
		Create a local response normalization layer.

		Args:
			depth_radius (int): See `tf.nn.local_response_normalization`.
			bias (float): See `tf.nn.local_response_normalization`.
			alpha (float): See `tf.nn.local_response_normalization`.
			beta (float): See `tf.nn.local_response_normalization`.
			name (str): The name to use for any `tf.Tensor`s created.
		"""
		super().__init__(LocalResponseNormalizationLayer.TYPE)
		assert isinstance(depth_radius, int)
		assert depth_radius > 0
		assert bias > 0
		assert alpha > 0
		self._depth_radius = depth_radius
		self._bias = bias
		self._alpha = alpha
		self._beta = beta
		self._name = name

	def output(self, previous):
		"""
		Get the output of this layer.

		Args:
			previous (tf.Tensor): The output of the layer before this one. Must have shape `[n_batches, height, width, channels]`.

		Returns:
			A `tf.Tensor` representing the output of this layer.
		"""
		super().output(previous)
		pooling_fn = None
		return tf.nn.local_response_normalization(
			previous,
			depth_radius = self._depth_radius,
			bias = self._bias,
			alpha = self._alpha,
			beta = self._beta,
			name = self._name
		)

class BatchNormalizationLayer(BaseLayer):
	"""
	A batch normalization layer of a neural network.
	"""

	def __init__(self, axis, name):
		"""
		Create a batch normalization layer.

		Args:
			axis (int): The axis that should be normalized.
			name (str): The name to use for any tensors created.
		"""
		self._axis = axis
		self._name = name

	def output(self, previous):
		"""
		Get the output of this layer.

		Args:
			previous (tf.Tensor): The output of the layer before this one. Must
				have shape `[n_batches, height, width, channels]`.

		Returns:
			A `tf.Tensor` representing the output of this layer.
		"""
		super().output(previous)
		return tf.layers.batch_normalization(previous, axis=self._axis,
			name=self._name)

class PoolingLayer(BaseLayer):
	"""
	A pooling layer of a neural network.
	"""

	TYPE = "POOL"
	POOLING_TYPES = set(["MAX", "AVG"])

	def __init__(self, pooling_type, window_side, window_stride, padding_method, name):
		"""
		Create a pooling layer.

		Args:
			pooling_type (str): The type for the pooling, either `"MAX"` or `"AVG"`.
			window_side (int): The side length of the sliding window for the pooling, in pixels.
			window_stride (int): The stride length of the sliding window for the pooling, in pixels.
			padding_method (str): The name of the padding method to use.
			name (str): The name to use for any `tf.Tensor`s created.
		"""
		super().__init__(PoolingLayer.TYPE)
		assert pooling_type in PoolingLayer.POOLING_TYPES
		assert isinstance(window_side, int)
		assert window_side >= 1
		assert isinstance(window_stride, int)
		assert window_stride >= 1
		assert padding_method in ConvolutionalLayer.PADDING_METHODS
		self._pooling_type = pooling_type
		self._window_side = window_side
		self._window_stride = window_stride
		self._padding_method = padding_method
		self._name = name

	def output(self, previous):
		"""
		Get the output of this layer.

		Args:
			previous (tf.Tensor): The output of the layer before this one. Must have shape `[n_batches, height, width, channels]`.

		Returns:
			A `tf.Tensor` representing the output of this layer.
		"""
		super().output(previous)
		pooling_fn = None
		if self._pooling_type == "MAX":
			pooling_fn = tf.layers.max_pooling2d
		elif self._pooling_type == "AVG":
			pooling_fn = tf.layers.average_pooling2d
		else:
			raise ValueError("invalid pooling type %s" % self._pooling_type)
		return pooling_fn(
			previous,
			self._window_side,
			self._window_stride,
			padding=self._padding_method,
			data_format=BaseLayer.DATA_FORMAT,
			name=self._name
		)

class TransferLayer(BaseLayer):
	"""
	A transfer layer of a neural network.
	"""

	TYPE = "TRAN"

	def __init__(self, transfer_fn, name):
		"""
		Create a transfer layer.

		Args:
			transfer_fn (func(tf.Tensor, str) -> tf.Tensor): The transfer function. The input
				`tf.Tensor` will have dtype `tf.float32` and the returned tensor must as well.
				The second argument gives the name to use for any `tf.Tensor`s created.
			name (str): The name to use for any `tf.Tensor`s created.
		"""
		super().__init__(TransferLayer.TYPE)
		self._transfer_fn = transfer_fn
		self._name = name

	def output(self, previous):
		"""
		Get the output of this layer.

		Args:
			previous (tf.Tensor): The output of the layer before this one. Must have shape `[n_batches, height, width, channels]`.

		Returns:
			A `tf.Tensor` representing the output of this layer.
		"""
		super().output(previous)
		return self._transfer_fn(previous, self._name)

class FlatLayer(BaseLayer):
	"""
	A flattening layer of a neural network.
	"""

	TYPE = "FLAT"

	def __init__(self, name):
		"""
		Create a flattening layer.

		Args:
			name (str): The name to use for any `tf.Tensor`s created.
		"""
		super().__init__(FlatLayer.TYPE)
		self._name = name

	def output(self, previous):
		"""
		Get the output of this layer.

		Args:
			previous (tf.Tensor): The output of the layer before this one. Must have shape `[n_batches, height, width, channels]`.

		Returns:
			A `tf.Tensor` representing the output of this layer.
		"""
		super().output(previous)
		return tf.contrib.layers.flatten(previous, outputs_collections=None, scope=None)

class FullLayer(BaseLayer):
	"""
	A fully-connected layer of a neural network.
	"""

	TYPE = "FULL"
	ACTIVATION = None
	USE_BIAS = True
	ACTIVITY_REGULARIZER = None
	TRAINABLE = True
	REUSE = False

	def __init__(self, size, weight_init, bias_init, weight_reg=None,
		bias_reg=None, name=None):
		"""
		Create a fully-connected layer.

		Args:
			size (int): The number of neurons for this layer. The output will have shape `[n_batches, size]`.
			weight_init (like return value of
				tf.contrib.keras.initializers.Initializer): An initializer for
				the weights.
			bias_init (like return value of
				tf.contrib.keras.initializers.Initializer): An initializer for
				the biases.
			weight_reg (like return value of tf.contrib.layers.l2_regularizer):
				A regularizer for the weights. If omitted or None, then no
				regularization is applied to the weights.
			bias_reg (like return value of tf.contrib.layers.l2_regularizer):
				A regularizer for the biases. If omitted or None, then no
				regulatization is applied to the biases.
			name (str): The name to use for any `tf.Tensor`s created.
		"""
		super().__init__(FullLayer.TYPE)
		assert size >= 1
		assert isinstance(weight_init,
			tf.contrib.keras.initializers.Initializer)
		assert isinstance(bias_init, tf.contrib.keras.initializers.Initializer)
		self._size = size
		self._weight_init = weight_init
		self._bias_init = bias_init
		self._weight_reg = weight_reg
		self._bias_reg = bias_reg
		self._name = name

	def output(self, previous):
		"""
		Get the output of this layer.

		Args:
			previous (tf.Tensor): The output of the layer before this one. Must have shape `[n_batches, size]`.

		Returns:
			A `tf.Tensor` representing the output of this layer.
		"""
		super().output(previous)
		return tf.layers.dense(
			previous,
			self._size,
			activation=FullLayer.ACTIVATION,
			use_bias=FullLayer.USE_BIAS,
			kernel_initializer=self._weight_init,
			bias_initializer=self._bias_init,
			kernel_regularizer=self._weight_reg,
			bias_regularizer=self._bias_reg,
			activity_regularizer=FullLayer.ACTIVITY_REGULARIZER,
			trainable=FullLayer.TRAINABLE,
			name=self._name,
			reuse=FullLayer.REUSE
		)

class DropoutLayer(BaseLayer):
	"""
	A dropout layer of a neural network.
	"""

	TYPE = "DROP"

	def __init__(self, keep_prob, noise_shape, seed, name):
		"""
		Create a fullly-connected layer.

		Args:
			keep_prob (float): See `tf.nn.dropout`.
			noise_shape (tuple of int): See `tf.nn.dropout`.
			seed (int): See `tf.nn.dropout`.
			name (str): The name to use for any `tf.Tensor`s created.
		"""
		super().__init__(DropoutLayer.TYPE)
		assert keep_prob > 0
		assert noise_shape is None or len(noise_shape) == 4
		assert noise_shape is None or all(isinstance(x, int) for x in noise_shape)
		assert noise_shape is None or all(x > 0 for x in noise_shape)
		self._keep_prob = keep_prob
		self._noise_shape = noise_shape
		self._seed = seed
		self._name = name

	def output(self, previous):
		"""
		Get the output of this layer.

		Args:
			previous (tf.Tensor): The output of the layer before this one. Must have shape `[n_batches, size]`.

		Returns:
			A `tf.Tensor` representing the output of this layer.
		"""
		super().output(previous)
		return tf.nn.dropout(
			previous,
			keep_prob = self._keep_prob,
			noise_shape = self._noise_shape,
			seed = self._seed,
			name = self._name
		)
