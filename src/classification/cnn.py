import tensorflow as tf

class Layer(object):
	TYPES = set(["CONV", "POOL", "TRAN", "FLAT", "FULL"])
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
		assert layer_type in Layer.TYPES
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
		assert isinstance(previous, tf.Tensor)

class ConvolutionalLayer(Layer):
	"""
	A convolutional layer of a neural network.
	"""

	TYPE = "CONV"
	PADDING_METHODS = set(["VALID", "SAME"])
	DILATION_RATE = 1
	ACTIVATION = None
	USE_BIAS = True
	KERNEL_INITIALIZER = tf.contrib.keras.initializers.glorot_normal()
	BIAS_INITIALIZER = tf.zeros.initializer()
	KERNEL_REGULARIZER = tf.contrib.keras.regularizers.l2()
	BIAS_REGULARIZER = tf.contrib.keras.regularizers.l2()
	ACTIVITY_REGULARIZER = None
	TRAINABLE = True
	REUSE = False

	def __init__(self, n_filters, window_side, window_stride, padding_method, name):
		"""
		Create a convolutional layer.

		Args:
			n_filters (int): Number of filters.
			window_side_length (int): The side length of the sliding window for the filters, in pixels. Must be odd.
			stride_length (int): The stride length of the sliding window for the filters, in pixels.
			padding_method (str): The name of the padding method to use.
			name (str): The name to use for any `tf.Tensor`s created.
		"""
		super().__init__(ConvolutionalLayer.TYPE)
		assert isinstance(n_filters, int)
		assert n_filters >= 1:
		assert isinstance(window_side, int)
		assert window_side >= 1
		assert window_side % 2 == 1:
		assert isinstance(window_stride, int)
		assert window_stride >= 1
		assert padding_method in ConvolutionalLayer.PADDING_METHODS
		self._n_filters = n_filters
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
		return tf.layers.conv2d(
			previous,
			self._n_filters,
			self._window_side,
			stides=self._window_stride,
			padding=self._padding_method,
			data_format=Layer.DATA_FORMAT,
			dilation_rate=ConvolutionalLayer.DILATION_RATE,
			activation=ConvolutionalLayer.ACTIVATION,
			use_bias=ConvolutionalLayer.USE_BIAS,
			kernel_initializer=ConvolutionalLayer.KERNEL_INITIALIZER,
			bias_initializer=ConvolutionalLayer.BIAS_INITIALIZER,
			kernel_regularizer=ConvolutionalLayer.KERNEL_REGULARIZER,
			bias_regularizer=ConvolutionalLayer.BIAS_REGULARIZER,
			activity_regularizer=ConvolutionalLayer.ACTIVITY_REGULARIZER,
			trainable=ConvolutionalLayer.TRAINABLE,
			name=self._name,
			resuse=ConvolutionalLayer.REUSE
		)

class PoolingLayer(Layer):
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
			window_side (int): The side length of the sliding window for the pooling, in pixels. Must be odd.
			window_stride (int): The stride length of the sliding window for the pooling, in pixels.
			padding_method (str): The name of the padding method to use.
			name (str): The name to use for any `tf.Tensor`s created.
		"""
		super().__init__(PoolingLayer.TYPE)
		assert pooling_type in PoolingLayer.POOLING_TYPES
		assert isinstance(window_side, int)
		assert window_side >= 1
		assert window_side % 2 == 1
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
			data_format=Layer.DATA_FORMAT,
			name=self._name
		)

class TransferLayer(Layer):
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

class FlatLayer(Layer):
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
		old_shape = tf.shape(previous, self._name)
		new_shape = tf.concat(
			[old_shape[0], old_shape[1] * old_shape[1] * old_shape[2]],
			axis=0,
			name=self._name
		)
		return tf.reshape(previous, new_shape, name=self._name)
