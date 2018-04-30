"""Test a PinnedSVM on the ``pinned`` dataset.

Run ``python test_svm.py -h`` to see usage details.
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
from cell_counting import dataset, visualization, postprocess

# ===============================
# Import from the Python library.
# ===============================
from argparse import ArgumentParser

# =================================
# Import from third-party packages.
# =================================
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from skimage import feature, color
import imageio
from sklearn.externals import joblib


def hog(images):
    hogs = []
    for i in tqdm(range(images.shape[0]), desc="Computing HOG",
                  unit="images"):
        image = color.rgb2grey(images[i, ...])
        hogs.append(feature.hog(image, orientations=9,
                                pixels_per_cell=(8, 8),
                                cells_per_block=(3, 3),
                                block_norm="L2-Hys", feature_vector=True))
    return np.stack(hogs, axis=0)


def discretize(counts):
    BINS = [5, 10, 20]

    for i in range(counts.shape[0]):
        for bin, bound in enumerate(BINS + [float("inf")]):
            if counts[i] <= bound:
                counts[i] = bin
                break


def save_image(path, image):
    RGB_MAX = 255

    os.makedirs(os.path.split(path)[0], exist_ok=True)
    image = image.astype(np.float64)
    image = image - np.amin(image)
    image = image * RGB_MAX / np.amax(image)
    image = image.astype(np.uint8)
    imageio.imwrite(path, image)


if __name__ == "__main__":
    # ===============================
    # Process command-line arguments.
    # ===============================
    parser = ArgumentParser(
        description='Test a PinnedSVM on the "pinned" dataset.')
    parser.add_argument("-pinned", type=str, required=False,
                        default="preprocess_pinned",
                        help="A path to a directory containing the output of "
                             "preprocess_pinned.py.")
    parser.add_argument("-tvsvm", type=str, required=False,
                        default="train_validate_svm",
                        help="A path to a directory containing the output of "
                             "train_validate_svm.py.")
    parser.add_argument("-out", type=str, required=False, default="test_svm",
                        help="A path to a directory in which to save output."
                             " Will be created if nonexistent.")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # =============
    # Load model.
    # =============
    with tqdm(desc="Loading model", total=1, unit="models") as prog:
        path = os.path.join(args.tvsvm, "model_save.pkl")
        model = joblib.load(path)
        prog.update(1)

    # ==============
    # Load datasets.
    # ==============
    with tqdm(desc="Loading datasets", total=2, unit="datasets") as prog:
        path = os.path.join(args.pinned, "pinned_train")
        train = dataset.Dataset(path)
        prog.update(1)

        path = os.path.join(args.pinned, "pinned_test")
        test = dataset.Dataset(path)
        prog.update(1)

    # ===========
    # Test model.
    # ===========
    with tqdm(desc="Determining normalization factor", total=1,
              unit="normalization factors") as prog:
        train_inputs, train_outputs = train.get_all()
        train_inputs = hog(train_inputs)
        norm_factor = np.amax(train_inputs)
        prog.update(1)

    with tqdm(desc="Preprocessing test dataset", total=1,
              unit="datasets") as prog:
        test_inputs, test_outputs = test.get_all()
        test_inputs_hog = hog(test_inputs)
        test_inputs_hog /= norm_factor
        discretize(test_outputs)
        prog.update(1)

    with tqdm(desc="Testing model", total=1, unit="datasets") as prog:
        predictions = model.predict(test_inputs_hog)
        prog.update(1)

    # ======================
    # Make confusion matrix.
    # ======================
    with tqdm(desc="Making confusion matrix", total=1,
              unit="confusion matrices") as prog:
        conf = metrics.confusion_matrix(test_outputs, predictions)
        path = os.path.join(args.out, "confusion_matrix.svg")
        visualization.plot_confusion_matrix(conf,
            "Confusion Matrix on Test Data", 6, 6, path)
        prog.update(1)

    # ===============
    # Make HOG plots.
    # ===============
    HOG_PLOTS = 3

    for i in tqdm(range(HOG_PLOTS), desc="Making HOG plots", unit="HOG plots"):
        image = color.rgb2grey(test_inputs[i, ...])
        hog_image = feature.hog(image, orientations=9,
                                pixels_per_cell=(8, 8), cells_per_block=(3, 3),
                                block_norm="L2-Hys", visualise=True,
                                feature_vector=True)[1]
        path = os.path.join(args.out, "hog", "{0}.png".format(i))
        save_image(path, hog_image)

    # =====================================
    # Make confidence cutoff-related plots.
    # =====================================
    with tqdm(desc="Testing model for probabilities", total=1,
              unit="models") as prog:
        probs = model.predict_proba(test_inputs_hog)
        cutoffs, accs, props = \
            postprocess.confidence_cutoff_analysis(probs, test_outputs)
        prog.update(1)

    with tqdm(desc="Making confidence cutoff plots", total=2,
              unit="plots") as prog:
        path = os.path.join(args.out, "accuracy_vs_cutoff.svg")
        visualization.plot_line(cutoffs, accs,
            "Effect of Confidence Level Cutoff on Prediction Reliability",
            "confidence level cutoff",
            "accuracy of predictions meeting cutoff", 4, 10, path=path)
        prog.update(1)

        path = os.path.join(args.out, "proportion_vs_cutoff.svg")
        visualization.plot_line(cutoffs, props,
            "Effect of Confidence Level Cutoff on Prediction Usability",
            "confidence level cutoff",
            "proportion of predictions meeting cutoff", 4, 10, path=path)
        prog.update(1)
