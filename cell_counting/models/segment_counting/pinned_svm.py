from multiprocessing.dummy import Pool

import psutil
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn import metrics
import numpy as np
from skimage import feature, color

class PinnedSVM(object):
    """An SVM for classifying the density of a pinned plate well."""

    _BYTES_PER_MB = 1_000_000
    _DEFAULT_HYPER_SET = {"cache_size": psutil.virtual_memory().available /
                                        _BYTES_PER_MB * 0.5,
                          "class_weight": "balanced"}

    def __init__(self, hyper_sets, bins):
        """Create a pinned SVM.

        Args:
            hyper_sets (list(dict)): A `list` of hyperparameter sets. A
                hyperparameter set is a `dict`. Its keys are `str`s which are
                some of the parameter names for `sklearn.svm.SVC`. Its values
                are the corresponding arguments to be used in creating a
                `sklearn.svm.SVC`.
            bins (list(int)): Bin boundaries (inclusive maximums) for
                discretizing counts.
        """
        self._hyper_sets = hyper_sets
        self._bins = bins
        self._model = None

    def train(self, train, valid):
        """Train this pinned SVM.

        Args:
            train (dataset.Dataset): The training dataset, used to train
                SVMs. Outputs are assumed to be counts.
            valid (dataset.Dataset): The validation dataset, used to assess
                SVMs and select the best. Outputs are assumed to be counts.

        Returns:
            list(tuple): A `list` of results. Each result is a `tuple` of 3
                elements. The first is a hyperparameter set. The second is
                the trained model's F-beta score on ``valid``. The third is the
                trained model.
        """
        def hog(images):
            hogs = []
            for i in tqdm(range(images.shape[0]), desc="Computing HOG",
                          unit="images"):
                image = color.rgb2grey(images[i, ...])
                hogs.append(feature.hog(image, orientations=9,
                    pixels_per_cell=(8, 8), cells_per_block=(3, 3),
                    block_norm="L2-Hys", feature_vector=True))
            return np.stack(hogs, axis=0)

        def discretize(counts):
            for i in range(counts.shape[0]):
                for bin, bound in enumerate(self._bins + [float("inf")]):
                    if counts[i] <= bound:
                        counts[i] = bin
                        break

        def check_hyper(hyper_set, hyper):
            if hyper not in hyper_set:
                raise ValueError("hyperparameter set does not supply a value "
                                 " for '{0:s}'".format(hyper))

        def evaluate_hyper_set(hyper_set):
                check_hyper(hyper_set, "C")
                check_hyper(hyper_set, "kernel")
                if hyper_set["kernel"] == "poly":
                    check_hyper(hyper_set, "degree")
                    check_hyper(hyper_set, "gamma")
                    check_hyper(hyper_set, "coef0")
                elif hyper_set["kernel"] == "rbf":
                    check_hyper(hyper_set, "gamma")
                elif hyper_set["kernel"] == "sigmoid":
                    check_hyper(hyper_set, "gamma")
                    check_hyper(hyper_set, "coef0")
                else:
                    raise ValueError("hyperparameter set supplies unsupported"
                                     " value for 'kernel'")
                defaults = self._DEFAULT_HYPER_SET.copy()
                defaults.update(hyper_set)
                hyper_set = defaults
                model = SVC(**hyper_set)
                model.fit(train_inputs, train_outputs)
                predicted = model.predict(valid_inputs)
                fscore = metrics.precision_recall_fscore_support(
                    valid_outputs, predicted, average="weighted")[2]
                return hyper_set, fscore, model

        train_inputs, train_outputs = train.get_all()
        train_inputs = hog(train_inputs)
        norm_factor = np.amax(train_inputs)
        train_inputs /= norm_factor
        discretize(train_outputs)
        valid_inputs, valid_outputs = valid.get_all()
        valid_inputs = hog(valid_inputs)
        valid_inputs /= norm_factor
        discretize(valid_outputs)

        with Pool(psutil.cpu_count(logical=False)) as pool:
            results = pool.imap(evaluate_hyper_set, self._hyper_sets, 1)
            results = tqdm(results, desc="Train models",
                           unit="hyperparameter sets",
                           total=len(self._hyper_sets))
            results = list(results)
        return results
