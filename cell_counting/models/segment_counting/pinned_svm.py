import psutil
from sklearn.svm import SVC


class PinnedSVM(object):
    _BYTES_PER_MB = 1_000_000
    _DEFAULT_MEMORY_PROP = 0.5
    _DEFAULT_HYPERS = {"class_weight": "balanced", "probability": True}

    def __init__(self, **hypers):
        def check_hypers(hypers, name):
            if name not in hypers:
                raise ValueError(
                    "hyperparameter '{0:s}' not supplied".format(name))
        check_hypers(hypers, "C")
        check_hypers(hypers, "kernel")
        if hypers["kernel"] == "poly":
            check_hypers(hypers, "degree")
            check_hypers(hypers, "gamma")
            check_hypers(hypers, "coef0")
        elif hypers["kernel"] == "rbf":
            check_hypers(hypers, "gamma")
        elif hypers["kernel"] == "sigmoid":
            check_hypers(hypers, "gamma")
            check_hypers(hypers, "coef0")
        else:
            raise ValueError("unsupported value supplied for 'kernel'")
        all_hypers = self._DEFAULT_HYPERS.copy()
        all_hypers["cache_size"] = (psutil.virtual_memory().available /
            self._BYTES_PER_MB * self._DEFAULT_MEMORY_PROP)
        all_hypers.update(hypers)
        self._model = SVC(**all_hypers)

    def train(self, inputs, outputs):
        self._model.fit(inputs, outputs)

    def predict(self, inputs):
        return self._model.predict(inputs)

    def predict_probs(self, inputs):
        return self._model.predict_proba(inputs)

