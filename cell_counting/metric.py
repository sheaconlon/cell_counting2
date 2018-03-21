from . import visualization, utilities

import os

import tensorflow as tf
import numpy as np


class BaseMetric(object):
    """A metric. Extracts some information about a `model.BaseModel`.
        Remembers its past evaluations."""

    def __init__(self, save_path):
        """Create a metric.

        Args:
            save_path (str): A path to a directory in which to save (or from
                which to load) the results of this metric's past evaluations.
        """
        self._save_path = save_path

    def evaluate(self, model):
        """Evaluate this metric on a model."""
        raise NotImplementedError

    def _write_save_file(self, basename, data):
        path = os.path.join(self._save_path, basename + ".npy")
        os.makedirs(self._save_path, exist_ok=True)
        np.save(path, data)

    def _read_save_file(self, basename):
        path = os.path.join(self._save_path, basename + ".npy")
        try:
            return np.load(path)
        except IOError:
            return None

    def _save_file_path(self, basename):
        return os.path.join(self._save_path, basename + ".npy")


class ConfusionMatrixMetric(BaseMetric):
    """A confusion matrix metric."""

    def __init__(self, save_path, data_fn, num_classes):
        """Create a confusion matrix metric.

        Args:
            save_path (str): See `BaseMetric`.
            data_fn (func): A data function. When called, returns a batch of
                data to test a model with. The batch is as a ``tuple`` of
                ``numpy.ndarray``s. The first array contains the inputs and
                the second array contains the outputs.
            num_classes (int): The number of classes that exist. It will be
                assumed that the classes are ``0, 1, ..., num_classes - 1``.
        """
        assert num_classes > 0, "argument num_classes must be greater than 0"

        super().__init__(save_path)

        self._data_fn = data_fn
        self._num_classes = num_classes

        self._train_steps = self._read_save_file("train_steps")
        self._confusion_matrices = self._read_save_file("confusion_matrices")

    def evaluate(self, model):
        """Evaluate the confusion matrix of a model.

        Args:
            model (model.BaseModel): The model. Must be a classifier, whose
                output is an array of per-class scores.

        Returns:
            An array of size (num_classes, num_classes) where the value at
                (i, j) is the number of times that the model predicted class
                i when the true class was class j.
        """
        inputs, actual = self._data_fn()
        actual = np.argmax(actual, axis=1)
        predicted = model.predict(inputs)
        predicted = np.argmax(predicted, axis=1)
        matrix = tf.confusion_matrix(tf.constant(actual),
                                     tf.constant(predicted),
                                     num_classes=self._num_classes)
        matrix = utilities.tensor_eval(matrix)
        matrix = np.transpose(matrix)
        train_steps = np.array([model.get_global_step()])
        confusion_matrices = matrix[np.newaxis, ...]
        if self._train_steps is not None:
            arrays = (self._train_steps, train_steps)
            train_steps = np.concatenate(arrays, axis=0)
            arrays = (self._confusion_matrices, confusion_matrices)
            confusion_matrices = np.concatenate(arrays, axis=0)
        self._train_steps = train_steps
        self._confusion_matrices = confusion_matrices
        self._write_save_file("train_steps", self._train_steps)
        self._write_save_file("confusion_matrices", self._confusion_matrices)
        return matrix

    def plot(self, title, height, width, path=None):
        """Plot the most recent confusion matrix of the model."""
        visualization.plot_confusion_matrix(self._confusion_matrices[-1, ...],
                                            title, height, width, path=path)


class LossMetric(BaseMetric):
    """A loss metric."""

    def __init__(self, save_path, data_fns, loss_fn):
        """Create a loss metric.

        Args:
            save_path (str): See ``BaseMetric``.
            data_fns (list(func)): A list of data functions. When called,
                a data function returns a batch of data to test a model with.
                The batch is as a ``tuple`` of ``numpy.ndarray``s. The first
                array contains the inputs and the second array contains the
                outputs.
            loss_fn (func): A loss function. Takes as arguments the correct
                outputs and the predicted outputs for a batch. Returns a scalar
                representing the loss for that batch.
        """
        super().__init__(save_path)

        self._data_fns = data_fns
        self._loss_fn = loss_fn

        self._train_steps = self._read_save_file("train_steps")
        self._losses = self._read_save_file("losses")

    def evaluate(self, model):
        """Evaluate the loss of a model.

        Args:
            model (model.BaseModel): The model.

        Returns:
            (numpy.ndarray) The losses. An array with shape
                ``(len(data_fns))``, whose ``i``-th element is the loss of
                ``model`` under ``data_fns[i]``. See ``data_fns`` in `__init__`.
        """
        losses = np.empty(len(self._data_fns))
        for i, data_fn in enumerate(self._data_fns):
            inputs, actual = data_fn()
            predicted = model.predict(inputs)
            loss = self._loss_fn(actual, predicted)
            assert len(loss.shape) == 0, \
                "The return value of a loss function – the loss – must be a " \
                "scalar."
            losses[i] = loss
        train_steps = np.array([model.get_global_step()])
        all_losses = losses[np.newaxis, ...]
        if self._train_steps is not None:
            arrays = (self._train_steps, train_steps)
            train_steps = np.concatenate(arrays, axis=0)
            arrays = (self._losses, all_losses)
            all_losses = np.concatenate(arrays, axis=0)
        self._train_steps = train_steps
        self._losses = all_losses
        self._write_save_file("train_steps", self._train_steps)
        self._write_save_file("losses", self._losses)
        return losses

    def plot(self, title, x_lab, y_lab, data_fn_labels, height, width,
             path=None):
        """Plot how the loss of the model under each of the ``loss_fns`` has
            varied over training."""
        sets_of_ys = [self._losses[i, ...]
                      for i in range(self._losses.shape[0])]
        visualization.plot_lines(self._train_steps, sets_of_ys, title,
                                 x_lab, y_lab, data_fn_labels, height, width,
                                 path=path)


class AccuracyMetric(LossMetric):
    """An accuracy metric."""

    def __init__(self, save_path, data_fns):
        """Create an accuracy metric.

        Args:
            save_path (str): See `BaseMetric`.
            data_fns (func): See `LossMetric`.
        """
        def loss_fn(actual, predicted):
            actual = np.argmax(actual, axis=1)
            predicted = np.argmax(predicted, axis=1)
            return np.mean(np.equal(actual, predicted))
        super().__init__(save_path, data_fns, loss_fn)
