from . import visualization

import time

import tensorflow as tf
import numpy as np

class BaseMetric(object):
	"""A metric."""

	def __init__(self):
		"""Create a metric."""
		self._examples_seen = []
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
			(tuple of lists): In the first list, the i-th element is the number
				of examples seen at the i-th evaluation. In the second
				list, the i-th element is the value of this metric during the
				i-th evaluation.
		"""
		return self._examples_seen, self._results

	def _record(self, model, result):
		self._examples_seen.append(
			model.get_global_step() * model.get_batch_size())
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
		self._record(model, conf_mtx)
		return conf_mtx

	def plot(self, title, height, width):
		"""Plot this metric."""
		examples_seen, conf_mtxs = self.get_results()
		visualization.plot_confusion_matrix(conf_mtxs[-1], title, height, width)

class NonexclusiveConfusionMatrixMetric(ConfusionMatrixMetric):
	"""A nonexclusive confusion matrix metric.

	A nonexclusive confusion matrix views a model as producing a prediction
		which is a probability distribution over classes. Instead of tallying
		one prediction for the class of maximal score/probability as in normal
		confusion matrices, it tallies a fractional prediction for each class.
		It then normalizes the entire confusion matrix so that its entries sum
		to the number of examples, as in normal confusion matrices. Note that
		to the calculate the probability distribution over classes, the scores
		are passed through softmax.
	"""

	def __init__(self, batch, num_classes):
		"""Create a nonexclusive confusion matrix metric.

		Args:
			batch (tuple of np.ndarray): See ConfusionMatrixMetric.__init__.
			num_classes (int): See ConfusionMatrixMetric.__init__.
		"""
		super().__init__(batch, num_classes)

	def evaluate(self, model):
		"""Calculate the nonexclusive confusion matrix of a model. Also records
			the result.

		Args:
			model (model.BaseModel): The model. Must be a classifier, whose
				output is an array of per-class scores.

		Returns:
			An array of size (num_classes, num_classes) where the value at
				(i, j) is like the number of times that the model predicted
				class i when the true class was class j.
		"""
		inputs, outputs = self._batch
		scores = model.predict(inputs)
		with tf.Session().as_default():
			pred_dist = tf.nn.softmax(tf.constant(scores), dim=1).eval()
		mtx = np.zeros((self._num_classes, self._num_classes))
		for i in range(outputs.shape[0]):
			true_class = np.argmax(outputs[i, :])
			mtx[:, true_class] += pred_dist[i, :]
		mtx /= np.sum(mtx)
		mtx *= inputs.shape[0]
		mtx = np.around(mtx).astype(int)
		self._record(model, mtx)
		return mtx

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
		loss = self._loss_fn(outputs, pred_outputs)
		self._record(model, loss)
		return loss

	def plot(self, title, x_lab, y_lab, height, width, path=None):
		"""Plot this metric."""
		examples_seen, losses = self.get_results()
		visualization.plot_line(examples_seen, losses, title, x_lab, y_lab,
								height, width, path)

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
		self._record(model, counts)
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
		self._record(model, thpt)
		return thpt
