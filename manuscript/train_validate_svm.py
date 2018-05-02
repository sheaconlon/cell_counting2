"""Train and validate some `PinnedSVM`s on the ``pinned`` dataset.

Run ``python train_validate_svm.py -h`` to see usage details.
"""

# ========================================
# Tell Python where to find cell_counting.
# ========================================
import sys, os

root_relative_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, root_relative_path)

# ==========================
# Import from cell_counting.
# ==========================
from cell_counting.models.segment_counting.pinned_svm import PinnedSVM
from cell_counting import dataset

# ===============================
# Import from the Python library.
# ===============================
from argparse import ArgumentParser
from multiprocessing import Pool
import os

# =================================
# Import from third-party packages.
# =================================
import numpy as np
from tqdm import tqdm
from sklearn.externals import joblib
from sklearn import metrics


def train_validate_svm(hypers):
    os.sched_setaffinity(os.getpid(), range(os.cpu_count()))
    train_path = os.path.join(args.pinned, "pinned_train")
    train = dataset.Dataset(train_path)
    train_images, train_classes = train.get_all()
    svm = PinnedSVM(**hypers)
    svm.train(train_images, train_classes)
    valid_path = os.path.join(args.pinned, "pinned_valid")
    valid = dataset.Dataset(valid_path)
    valid_images, valid_classes = valid.get_all()
    pred_valid_classes = svm.predict(valid_images)
    valid_accuracy = metrics.accuracy_score(valid_classes,
                                            pred_valid_classes)
    return valid_accuracy, svm


if __name__ == "__main__":
    # ===============================
    # Process command-line arguments.
    # ===============================
    parser = ArgumentParser(
        description='Train and validate some PinnedSVMs on the "pinned"'
                    ' dataset.')
    parser.add_argument("-pinned", type=str, required=False,
                        default="preprocess_pinned",
                        help="A path to a directory containing the output of "
                             "preprocess_pinned.py.")
    parser.add_argument("-out", type=str, required=False,
                        default="train_validate_svm",
                        help="A path to a directory in which to save output."
                             " Will be created if nonexistent.")
    parser.add_argument("-cvals", type=int, required=False, default=32,
                        help="The number of values of C to try for each"
                             " combination of the other hyperparameters.")
    parser.add_argument("-processes", type=int, required=False, default=4,
                        help="The number of processes to spawn.")
    args = parser.parse_args()

    assert args.cvals > 0, "cvals should be positive"
    assert args.processes > 0, "processes should be positive"
    os.makedirs(args.out, exist_ok=True)

    # ================
    # Construct model.
    # ================
    HYPER_SETS = [{"kernel": "rbf", "gamma": "auto"},
                  {"kernel": "poly", "degree": 2, "gamma": "auto", "coef0": 0},
                  {"kernel": "poly", "degree": 3, "gamma": "auto", "coef0": 0},
                  {"kernel": "sigmoid", "gamma": "auto", "coef0": 0}]
    C_MIN = 1e-2
    C_MAX = 1e3

    hyper_sets = []
    for hypers in HYPER_SETS:
        for C in np.geomspace(C_MIN, C_MAX, num=args.cvals):
            hypers = hypers.copy()
            hypers["C"] = C
            hyper_sets.append(hypers)
    with tqdm(desc="Training/validating models", unit="model",
              total=len(hyper_sets)) as prog:
        with Pool(processes=args.processes, maxtasksperchild=1) as pool:
            results = pool.imap(train_validate_svm, hyper_sets, chunksize=1)
            best_accuracy, best_svm = float("-inf"), None
            for accuracy, svm in results:
                if accuracy > best_accuracy:
                    best_accuracy, best_svm = accuracy, svm
                prog.update(1)

    # ===========
    # Save model.
    # ===========
    with tqdm(desc="Saving model", unit="model", total=1) as prog:
        path = os.path.join(args.out, "model_save.pkl")
        joblib.dump(best_svm, path)
        prog.update(1)
