import time

import tensorflow as tf

class Model(tf.estimator.Estimator):
	"""A model."""

	CONFIG = tf.estimator.RunConfig()
	CONFIG.replace(
		keep_checkpoint_every_n_hours=1,
		keep_checkpoint_max=float("inf"),
		log_step_count_steps=100,
		save_checkpoints_secs=60,
		tf_random_seed=42114
	)
	STEPS_AT_A_TIME = 1

	def __init__(self, loss_fn, eval_metric_fns, optimizer_fn, directory, name):
		"""
		Create a model.

		Args:
			loss_fn (func(tf.Tensor, tf.Tensor) -> tf.Tensor): The loss function. It accepts
				the correct outputs for a batch and the predicted outputs for a batch, both as
				`tf.Tensor`s with shape `[n_batches, output_dim]`.
			eval_metric_fns (dict(str: func(tf.Tensor, tf.Tensor) -> (tf.Tensor, tf.Tensor))):
				A dictionary from evaluation metric names to evaluation metric functions. Each
				evaluation metric function accepts the correct outputs for a batch and the
				predicted outputs for a batch, both as `tf.Tensor`s with shape `[n_batches, output_dim]`.
				See examples in `tf.metrics` for information on return values.
			optimizer_fn (func() -> tf.train.Optimizer): The optimizer function. It takes no
				arguments and returns a `tf.train.Optimizer`, which will be used to minimize the loss.
			directory (str): The path to the directory where this model's checkpoints should be saved.
			name (str): The name to use for any `tf.Tensor`s created.

		"""
		self._loss_fn = loss_fn
		self._eval_metric_fns = eval_metric_fns
		self._optimizer_fn = optimizer_fn
		self._name = name
		super().__init__(self._model_fn, model_dir=directory, config=Model.CONFIG, params=None)

	def _model_fn(self, features, labels, mode, params, config):
		predictions = self._output(features)
		if mode == tf.estimator.ModeKeys.PREDICT:
		    return tf.estimator.EstimatorSpec(
		        mode=mode,
		        predictions={"predictions": predictions}
		    )

		loss = self._loss_fn(labels, predictions)
		eval_metrics = {name: metric_fn(labels, predictions) for name, metric_fn in self._eval_metric_fns.items()}
		optimizer = self._optimizer_fn() # TODO: Pass params to the *fns.
		train_op = optimizer.minimize(
		    loss=loss,
		    global_step=tf.train.get_global_step()
		)
		return tf.estimator.EstimatorSpec(
		    mode=mode,
		    loss=loss,
		    train_op=train_op,
		    eval_metric_ops=eval_metrics
		)

	def _output(self, features):
		raise NotImplementedError

	def train_(self, data_fn):
		"""
		Train this model.

		Args:
			data_fn (func() -> tuple(tf.Tensor, tf.Tensor)): Function to supply training data.
				On each call, returns a different tuple of input data and correct output data.
				If it raises `StopIteration`, then the round of training stops.
		"""
		self.train(data_fn, hooks=None, steps=None, max_steps=None)

	def train_duration_(self, data_fn, duration):
		"""
		Train this model.

		Args:
			data_fn (func() -> tuple(tf.Tensor, tf.Tensor)): Function to supply training data.
				On each call, returns a different tuple of input data and correct output data.
				If it raises `StopIteration`, then the round of training stops.
			duration (int): The number of seconds to train for.
		"""
		start = time.time()
		while time.time() - start < duration:
			self.train(data_fn, hooks=None, steps=Model.STEPS_AT_A_TIME, max_steps=None)

	def evaluate_(self, data_fn):
		"""
		Evaluate this model.

		Args:
			data_fn (func() -> tuple(tf.Tensor, tf.Tensor)): Function to supply test data.
				On each call, returns a different tuple of input data and correct output data.
				If it raises `StopIteration`, then evaluation stops.

		Returns:
			Evaluation metrics for the given test data. A `(dict(str: tf.Tensor))`, where each
				entry corresponds to an evaluation metric function passed in upon construction.
		"""
		return self.evaluate(data_fn, hooks=None, checkpoint_path=None, name=self._name)

	def predict_(self, data_fn):
		"""
		Predict using this model.

		Args:
			data_fn (func() -> tf.Tensor): Function to supply prediction input data. On each call,
				returns different input data. If it raises `StopIteration`, then prediction stops.

		Returns:
			Predictions for the given prediction input data.
		"""
		return self.evaluate(data_fn, predict_keys=None, hooks=None, checkpoint_path=None)

class NeuralNet(Model):
	"""A neural network."""

	def __init__(self, layers, loss_fn, eval_metric_fns, optimizer_fn, directory, name):
		"""
		Create a neural network.

		Args:
			layers (list of `Layer`): The layers for the network.
			loss_fn (func(tf.Tensor, tf.Tensor) -> tf.Tensor): The loss function. It accepts
				the correct outputs for a batch and the predicted outputs for a batch, both as
				`tf.Tensor`s with shape `[n_batches, output_dim]`.
			eval_metric_fns (dict(str: func(tf.Tensor, tf.Tensor)) -> tf.Tensor): A dictionary from
				evaluation metric names to evaluation metric functions. Each evaluation metric function
				accepts the correct outputs for a batch and the predicted outputs for a batch, both as
				`tf.Tensor`s with shape `[n_batches, output_dim]`.
			optimizer_fn (func() -> tf.train.Optimizer): The optimizer function. It takes no
				arguments and returns a `tf.train.Optimizer`, which will be used to minimize the loss.
			directory (str): The path to the directory where this model's checkpoints should be saved.
			name (str): The name to use for any `tf.Tensor`s created.
		"""
		super().__init__(loss_fn, eval_metric_fns, optimizer_fn, directory, name)
		self._layers = layers

	def _output(self, features):
		previous = features
		for layer in self._layers:
			previous = layer.output(previous)
		return previous

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
		#assert type(previous) == tf.Tensor

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
	BIAS_INITIALIZER = tf.zeros_initializer()
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
		assert n_filters >= 1
		assert isinstance(window_side, int)
		assert window_side >= 1
		assert window_side % 2 == 1
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
			strides=self._window_stride,
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
			reuse=ConvolutionalLayer.REUSE
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
		return tf.contrib.layers.flatten(previous, outputs_collections=None, scope=None)

class FullLayer(Layer):
	"""
	A fullly-connected layer of a neural network.
	"""

	TYPE = "FULL"
	ACTIVATION = None
	USE_BIAS = True
	KERNEL_INITIALIZER = tf.contrib.keras.initializers.glorot_normal()
	BIAS_INITIALIZER = tf.zeros_initializer()
	KERNEL_REGULARIZER = tf.contrib.keras.regularizers.l2()
	BIAS_REGULARIZER = tf.contrib.keras.regularizers.l2()
	ACTIVITY_REGULARIZER = None
	TRAINABLE = True
	REUSE = False

	def __init__(self, size, name):
		"""
		Create a fullly-connected layer.

		Args:
			size (int): The number of neurons for this layer. The output will have shape `[n_batches, size]`.
			name (str): The name to use for any `tf.Tensor`s created.
		"""
		super().__init__(FullLayer.TYPE)
		assert size >= 1
		self._size = size
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
			kernel_initializer=FullLayer.KERNEL_INITIALIZER,
			bias_initializer=FullLayer.BIAS_INITIALIZER,
			kernel_regularizer=FullLayer.KERNEL_REGULARIZER,
			bias_regularizer=FullLayer.BIAS_REGULARIZER,
			activity_regularizer=FullLayer.ACTIVITY_REGULARIZER,
			trainable=FullLayer.TRAINABLE,
			name=self._name,
			reuse=FullLayer.REUSE
		)

