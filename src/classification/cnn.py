import tensorflow as tf

class Layer(object):
	TYPES = set(["CONV", "POOL", "TRAN", "FLAT", "FULL"])

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

	def output(self, previous):
		"""
		Get the output of this layer.

		Args:
			previous (tf.Tensor): The output of the layer before this one.
		"""
		raise NotImplementedError

class ConvolutionalLayer(Layer):
	"""
	A convolutional layer of a neural network.
	"""

	PADDING_METHODS = set(["VALID", "SAME"])
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
		assert isinstance(previous, tf.Tensor)
		return tf.layers.conv2d(
			previous,
			self._n_filters,
			self._window_side,
			self._window_stride,
			self._padding_method,
			ConvolutionalLayer.ACTIVATION,
			ConvolutionalLayer.USE_BIAS,
			ConvolutionalLayer.KERNEL_INITIALIZER,
			ConvolutionalLayer.BIAS_INITIALIZER,
			ConvolutionalLayer.KERNEL_REGULARIZER,
			ConvolutionalLayer.BIAS_REGULARIZER,
			ConvolutionalLayer.ACTIVITY_REGULARIZER,
			ConvolutionalLayer.TRAINABLE,
			name=self._name,
			ConvolutionalLayer.REUSE
		)
