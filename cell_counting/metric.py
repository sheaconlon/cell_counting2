from . import visualization

import time, os

import tensorflow as tf
import numpy as np


class BaseMetric(object):
    """A metric.

    A metric remembers its past evaluations by saving them to disk.
    """

    def __init__(self, save_path):
        """Create a metric.

        Args:
            save_path (str): A path to a directory in which to save this metric.
            figure_path (str): A path to a directory in which to save plots.
        """
        self._save_path = save_path
        self._init_save_dir()

    def evaluate(self, model):
        """Evaluate this metric of a model and save the result."""
        raise NotImplementedError

    def _init_save_dir(self):
        os.makedirs(self._save_path, exist_ok=True)
        if os.path.exists(self._save_file_path("save_exists")):
            return
        self._write_save_file("save_exists", True)

    def _save_file_path(self, basename):
        return os.path.join(self._save_path, basename + ".npy")

    def _write_save_file(self, basename, data):
        path = os.path.join(self._save_path, basename + ".npy")
        np.save(path, data)

    def _read_save_file(self, basename):
        path = os.path.join(self._save_path, basename + ".npy")
        try:
            return np.load(path)
        except IOError:
            return None


class ConfusionMatrixMetric(BaseMetric):
    """A confusion matrix metric."""

    def __init__(self, save_path, data, num_classes):
        """Create a confusion matrix metric.

        Args:
            save_path (str): See ``BaseMetric``.
            data (tuple of np.ndarray): A tuple of input data and output data to
                test models with.
            num_classes (int): The number of classes.
        """
        super().__init__(save_path)
        self._data = data
        self._num_classes = num_classes
        self._confusion_matrices = np.zeros((1, num_classes, num_classes),
                                            dtype=int)
        self._save()

    def evaluate(self, model):
        """Evaluate the confusion matrix of a model and save the result.

        Args:
            model (model.BaseModel): The model. Must be a classifier, whose
                output is an array of per-class scores.

        Returns:
            An array of size (num_classes, num_classes) where the value at
                (i, j) is the number of times that the model predicted class
                i when the true class was class j.
        """
        inputs, outputs = self._data
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
        self._confusion_matrices = np.concatenate(
            (self._confusion_matrices, conf_mtx[np.newaxis, ...]), axis=0)
        self._save()
        return conf_mtx

    def plot(self, title, height, width, path=None):
        """Plot this metric."""
        visualization.plot_confusion_matrix(self._confusion_matrices[-1, ...],
                                            title, height, width, path=path)

    def _save(self):
        self._write_save_file("confusion_matrices", self._confusion_matrices)


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

    def __init__(self, save_path, data, loss_fn):
        """Create a loss metric.

        Args:
            save_path (str): See ``BaseMetric``.
            data (tuple of np.ndarray): A tuple of input data and output data to
                test models with.
            loss_fn (func(np.ndarray, np.ndarray) -> float): A function
                that receives as arguments a batch of predicted outputs and
                correct outputs and returns a scalar representing the loss for
                this batch.
        """
        super().__init__(save_path)
        self._data = data
        self._loss_fn = loss_fn
        self._training_iterations = np.array([0])
        self._losses = np.array([float("inf")])

    def evaluate(self, model):
        """Evaluate the loss of a model and save the result.

        Args:
            model (model.BaseModel): The model.

        Returns:
            (float) The loss.
        """
        inputs, outputs = self._data
        pred_outputs = model.predict(inputs)
        loss = self._loss_fn(outputs, pred_outputs)
        training_iteration = model.get_global_step() * model.get_batch_size()
        self._training_iterations = np.concatenate((self._training_iterations,
        	np.array([training_iteration])), axis=0)
        self._losses = np.concatenate((self._losses, loss[np.newaxis,...]),
                                      axis=0)
       	self._save()
        return loss

    def plot(self, title, x_lab, y_lab, height, width, path=None):
        """Plot this metric."""
        visualization.plot_line(self._training_iterations, self._losses, title,
        						x_lab, y_lab, height, width, path=path)

    def _save(self):
        self._write_save_file("training_iterations", self._training_iterations)
        self._write_save_file("losses", self._losses)


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
        counts = np.zeros(2 * self._num_classes + 1)
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
