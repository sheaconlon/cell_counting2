"""Train and validate a PinnedSVM on the ``pinned`` dataset.

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
from cell_counting.models.segment_counting import pinned_svm
from cell_counting import dataset

# ===============================
# Import from the Python library.
# ===============================
from argparse import ArgumentParser

# =================================
# Import from third-party packages.
# =================================
import numpy as np
from tqdm import tqdm
from sklearn.externals import joblib

if __name__ == "__main__":
    # ===============================
    # Process command-line arguments.
    # ===============================
    parser = ArgumentParser(
        description='Train and validate a PinnedSVM on the "pinned" dataset.')
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
    args = parser.parse_args()

    assert args.cvals > 0, "cvals should be positive"
    os.makedirs(args.out, exist_ok=True)

    # ================
    # Construct model.
    # ================
    HYPER_SETS = [{"kernel": "rbf", "gamma": "auto"},
                  {"kernel": "poly", "degree": 2, "gamma": "auto", "coef0": 0},
                  {"kernel": "poly", "degree": 3, "gamma": "auto", "coef0": 0},
                  {"kernel": "sigmoid", "gamma": "auto", "coef0": 0}]
    TEST_P = 0.2
    VALID_P = 0.2
    C_MIN = 1e-2
    C_MAX = 1e3
    BINS = [5, 10, 20]

    hyper_sets = []
    for hyper_set in HYPER_SETS:
        for C in np.geomspace(C_MIN, C_MAX, num=args.cvals):
            hyper_set = hyper_set.copy()
            hyper_set["C"] = C
            hyper_sets.append(hyper_set)

    model = pinned_svm.PinnedSVM(hyper_sets, BINS)

    # ==============
    # Load datasets.
    # ==============
    with tqdm(desc="Loading datasets", total=2, unit="datasets") as prog:
        train_path = os.path.join(args.pinned, "pinned_train")
        train = dataset.Dataset(train_path)
        prog.update(1)

        valid_path = os.path.join(args.pinned, "pinned_valid")
        valid = dataset.Dataset(valid_path)
        prog.update(1)

    # ============
    # Train model.
    # ============
    model.train(train, valid)

    # ===========
    # Save model.
    # ===========
    with tqdm(desc="Saving model", total=1, unit="models") as prog:
        path = os.path.join(args.out, "model_save.pkl")
        joblib.dump(model._model, path)
        prog.update(1)
