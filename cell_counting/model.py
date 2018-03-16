import time, itertools

import tensorflow as tf
import numpy as np

class BaseModel(object):
	"""A model."""

	_CHKPT_MAX = 2
	_LOG_STEPS = 1000
	_TRAIN_STEPS = 100
	_SECS_PER_MIN = 60

	def __init__(self, save_dir, chkpt_save_interval, loss_fn,
			optimizer_factory):
		"""Create a model.

		Args:
			save_dir (str): The path of the directory to save the model in
				and/or load the model from.
			chkpt_save_interval (int): The number of minutes to wait between
				saving checkpoints into save_dir.
			loss_fn (func(tf.Tensor, tf.Tensor) -> tf.Tensor): Takes the correct
				outputs and the predicted outputs for a batch, both as tensors
				with shape (batch_size, ...). Returns a scalar tensor which is
				the loss for this batch.
			optimizer_factory (func(tf.Tensor) -> tf.train.Optimizer): Takes the
				global step and returns an optimizer which will be used to train
				the neural net.
		"""
		config = tf.estimator.RunConfig(model_dir=save_dir,
			save_checkpoints_secs=chkpt_save_interval*self._SECS_PER_MIN,
			keep_checkpoint_max=self._CHKPT_MAX,
			log_step_count_steps=self._LOG_STEPS)
		model_fn = self._make_model_fn(loss_fn, optimizer_factory)
		self._estimator = tf.estimator.Estimator(model_fn, config=config)
		self._global_step = 0

	def train(self, dataset, seconds):
		"""Train the model on some data.

		Args:
			dataset (dataset.Dataset): The dataset to train on.
			(int) seconds: The (approximate) number of seconds to train for.
		"""
		with self._set_up_tf() as session:
			start = time.time()
			batches = 0
			while True:
				data_fn = dataset.get_data_fn(self.get_batch_size(),
					self._TRAIN_STEPS)
				self._estimator.train(data_fn, steps=self._TRAIN_STEPS)
				batches += 1
				self._global_step += self._TRAIN_STEPS
				time_so_far = time.time() - start
				time_one_batch = time_so_far / batches
				if time_so_far + time_one_batch >= seconds:
					break
		session.close()

	def predict(self, inputs):
		"""Predict the outputs for some inputs.

		Args:
			inputs (np.ndarray): The inputs.

		Returns:
			(np.ndarray): The predicted outputs.
		"""
		with self._set_up_tf() as session:
			data_fn = tf.estimator.inputs.numpy_input_fn({"inputs":inputs},
				None, inputs.shape[0], 1, shuffle=False,
				queue_capacity=inputs.shape[0])
			return np.stack(list(itertools.islice(
				self._estimator.predict(data_fn), inputs.shape[0])), axis=0)
		session.close()

	def evaluate(self, metrics):
		"""Evaluate some metrics about the model.

		Args:
			(dict(str, metrics.BaseMetric)) metrics: A mapping from metric names
				to metrics.

		Returns:
			(dict(str, np.ndarray)): A mapping from metric names to the metric
				results.
		"""
		result = {name: metric.evaluate(self) for name, metric
				in metrics.items()}
		return result

	def get_global_step(self):
		"""Get the number of training iterations that this model has gone
			through.
		"""
		return self._global_step

	def _set_up_tf(self):
		tf.logging.set_verbosity(tf.logging.WARN)
		return tf.Session().as_default()

	def _make_model_fn(self, loss_fn, optimizer_factory):
		def model_fn(features, labels, mode, config):
			inputs = tf.cast(features["inputs"], dtype=tf.float32)
			if labels is not None:
				outputs = tf.cast(labels, dtype=tf.float32)
			regularize = (mode == tf.estimator.ModeKeys.TRAIN)
			predicted_outputs = self._tensor_predict(inputs, regularize)
			if mode == tf.estimator.ModeKeys.PREDICT:
			    return tf.estimator.EstimatorSpec(mode=mode, 
			    	predictions=predicted_outputs)
			loss = loss_fn(outputs, predicted_outputs)
			optimizer = optimizer_factory(tf.train.get_global_step())
			train_op = optimizer.minimize(loss=loss,
				global_step=tf.train.get_global_step())
			return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
				train_op=train_op)
		return model_fn

	def _tensor_predict(self, inputs, regularize):
		raise NotImplementedError

	def get_batch_size(self):
		raise NotImplementedError
