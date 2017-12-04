import itertools, time

import tensorflow as tf
import numpy as np

class BaseMetric(object):
	"""A metric."""

	def __init__(self):
		"""Create a metric."""
		self._steps = []
		self._results = []

	def evaluate(self, model):
		"""Evaluate a model. Also records the result.

		Args:
			model (model.BaseModel): The model.

		Returns:
			(tf.Tensor) The value of this metric for the model.
		"""
		raise NotImplementedError

	def get_results(self):
		"""Get the results for this metric from all past evaluations.

		Returns:
			(tuple of lists): In the first list, the i-th element is the value
				of the global step during the i-th evaluation. In the second
				list, the i-th element is the value of this metric during the
				i-th evaluation.
		"""
		return self._steps, self._results

	def _record(self, step, result):
		self._steps.append(step)
		self._results.append(result)

class ConfusionMatrixMetric(BaseMetric):
	"""A confusion matrix metric."""

	def __init__(self, batch, num_classes):
		"""Create a confusion matrix metric.

		Args:
			batch (tuple of np.ndarray): A batch of input and output data to
				test models with.
			num_classes (int): The number of classes.
		"""
		super().__init__()
		self._batch = batch
		self._num_classes = num_classes

	def evaluate(self, model):
		"""Calculate the confusion matrix of a model. Also records the result.

		Args:
			model (model.BaseModel): The model. Must be a classifier, whose
				output is an array of per-class scores.

		Returns:
			An array of size (num_classes, num_classes) where the value at
				(i, j) is the number of times that the model predicted class
				i when the true class was class j.
		"""
		inputs, outputs = self._batch
		pred_outputs = model.predict(inputs)
		outputs = np.argmax(outputs, axis=1)
		pred_outputs = np.argmax(pred_outputs, axis=1)
		with tf.Session().as_default():
			conf_mtx = tf.confusion_matrix(
				tf.constant(outputs),
				tf.constant(pred_outputs),
				num_classes=self._num_classes
			).eval()
		conf_mtx = np.transpose(conf_mtx)
		self._record(model.get_global_step(), conf_mtx)
		return conf_mtx

class LossMetric(BaseMetric):
	"""A loss metric."""

	def __init__(self, batch, loss_fn):
		"""Create a loss metric.

		Args:
			batch (tuple of np.ndarray): A batch of input and output data to
				test models with.
			loss_fn (func(np.ndarray, np.ndarray) -> float): A function
				that receives as arguments a batch of predicted outputs and
				correct outputs and returns a scalar representing the loss for
				this batch.
		"""
		super().__init__()
		self._batch = batch
		self._loss_fn = loss_fn

	def evaluate(self, model):
		"""Calculate the loss of a model. Also records the result.

		Args:
			model (model.BaseModel): The model.

		Returns:
			(float) The loss.
		"""
		inputs, outputs = self._batch
		pred_outputs = model.predict(inputs)
		loss = self._loss_fn(pred_outputs, outputs)
		self._record(model.get_global_step(), loss)
		return loss

class OffByCountMetric(BaseMetric):
	"""An off-by count metric."""

	def __init__(self, batch, num_classes):
		"""Create an off-by count metric.

		Args:
			batch (tuple of np.ndarray): A batch of input and output data to
				test models with.
			num_classes (int): The number of classes.
		"""
		super().__init__()
		self._batch = batch
		self._num_classes = num_classes

	def evaluate(self, model):
		"""Calculate the off-by counts of a model. Also records the result.

		Args:
			model (model.BaseModel): The model.

		Returns:
			(np.ndarray): An array with shape (2*self._num_classes + 1) where
				the entries are the number of times the model's prediction
				was off by -num_classes, ..., 0, ..., and num_classes,
				respectively.
		"""
		inputs, outputs = self._batch
		pred_outputs = model.predict(inputs)
		outputs = np.argmax(outputs, axis=1)
		pred_outputs = np.argmax(pred_outputs, axis=1)
		diffs = pred_outputs - outputs
		counts = np.zeros(2*self._num_classes + 1)
		for i in range(diffs.shape[0]):
			counts[diffs[i] + self._num_classes] += 1
		self._record(model.get_global_step(), counts)
		return counts

class PredictionThroughputMetric(BaseMetric):
	"""A prediction time metric."""

	_TRIALS = 10

	def __init__(self, batch):
		"""Create a prediction time metric.

		Args:
			batch (tuple of np.ndarray): A batch of input and output data to
				test models with.
		"""
		super().__init__()
		self._batch = batch

	def evaluate(self, model):
		"""Calculate the average prediction throughput of a model. Also records
			the result.

		Args:
			model (model.BaseModel): The model.

		Returns:
			(float): The average number of examples per second that the model
				can predict.
		"""
		inputs, _ = self._batch
		start = time.time()
		for _ in range(self._TRIALS):
			model.predict(inputs)
		thpt = self._TRIALS / (time.time() - start)
		self._record(model.get_global_step(), thpt)
		return thpt
