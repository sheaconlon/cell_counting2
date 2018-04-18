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
from cell_counting import dataset, visualization

# ===============================
# Import from the Python library.
# ===============================
from argparse import ArgumentParser
from shutil import rmtree

# =================================
# Import from third-party packages.
# =================================
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from skimage import feature, color
import imageio

if __name__ == "__main__":
    # ===============================
    # Process command-line arguments.
    # ===============================
    parser = ArgumentParser(
        description='Train a PinnedSVM on the "pinned" dataset.')
    parser.add_argument("-pinned", type=str, required=False,
                        default="preprocess_pinned",
                        help="A path to a directory containing the output of "
                             "preprocess_pinned.py.")
    parser.add_argument("-out", type=str, required=False, default="train_svm",
                        help="A path to a directory in which to save output."
                             " Will be created if nonexistent.")
    parser.add_argument("-cvals", type=int, required=False, default=16,
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
    BINS = [3, 6, 10, 15, 21]

    hyper_sets = []
    for hyper_set in HYPER_SETS:
        for C in np.geomspace(C_MIN, C_MAX, num=args.cvals):
            hyper_set = hyper_set.copy()
            hyper_set["C"] = C
            hyper_sets.append(hyper_set)

    hyper_sets = [{'kernel': 'sigmoid', 'gamma': 'auto', 'coef0': 0, 'C': 215.44346900318823}]
    model = pinned_svm.PinnedSVM(hyper_sets, BINS)

    # =============
    # Load dataset.
    # =============
    with tqdm(desc="Loading dataset", total=1, unit="datasets") as prog:
        pinned = dataset.Dataset("preprocess_pinned/pinned")
        prog.update(1)
    with tqdm(desc="Splitting dataset", total=2, unit="splits") as prog:
        train_valid, test = pinned.split(TEST_P, "tmp_pinned_train_valid",
                                         "tmp_pinned_test")
        prog.update(1)
        train, valid = pinned.split(VALID_P / (1-TEST_P), "tmp_pinned_train",
                                    "tmp_pinned_valid")
        prog.update(1)

    # ============
    # Train model.
    # ============
    results = model.train(train, valid)

    # =============
    # Plot results.
    # =============
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
        for i in range(counts.shape[0]):
            for bin, bound in enumerate(BINS + [float("inf")]):
                if counts[i] <= bound:
                    counts[i] = bin
                    break

    hyper_set, valid_fscore, model = max(results, key=lambda result: result[1])
    test_inputs, test_outputs = test.get_all()
    hog_test_inputs = hog(test_inputs)
    train_inputs, train_outputs = train.get_all()
    train_inputs = hog(train_inputs)
    norm_factor = np.amax(train_inputs)
    hog_test_inputs /= norm_factor
    discretize(test_outputs)
    # =============
    # TODO: MIGHT BE WRONG WAY
    # =============
    conf = metrics.confusion_matrix(test_outputs, model.predict(hog_test_inputs))
    path = os.path.join(args.out, "confusion_matrix.svg")
    visualization.plot_confusion_matrix(conf, "Confusion Matrix on Test Data",
                                        6, 6, path)

    def save_image(path, image):
        RGB_MAX = 255

        os.makedirs(os.path.split(path)[0], exist_ok=True)
        image = image.astype(np.float64)
        image = image - np.amin(image)
        image = image * RGB_MAX / np.amax(image)
        image = image.astype(np.uint8)
        imageio.imwrite(path, image)


    image = color.rgb2grey(test_inputs[0, ...])
    hog_image = feature.hog(image, orientations=9,
                    pixels_per_cell=(8, 8), cells_per_block=(3, 3),
                    block_norm="L2-Hys", visualise=True, feature_vector=True)[1]
    path = os.path.join(args.out, "hog_image1.png")
    save_image(path, hog_image)

    image = color.rgb2grey(test_inputs[1, ...])
    hog_image = feature.hog(image, orientations=9,
                            pixels_per_cell=(8, 8), cells_per_block=(3, 3),
                            block_norm="L2-Hys", visualise=True,
                            feature_vector=True)[1]
    path = os.path.join(args.out, "hog_image2.png")
    save_image(path, hog_image)

    image = color.rgb2grey(test_inputs[2, ...])
    hog_image = feature.hog(image, orientations=9,
                            pixels_per_cell=(8, 8), cells_per_block=(3, 3),
                            block_norm="L2-Hys", visualise=True,
                            feature_vector=True)[1]
    path = os.path.join(args.out, "hog_image3.png")
    save_image(path, hog_image)

    # =========
    # Clean up.
    # =========
    rmtree("tmp_pinned_train_valid")
    rmtree("tmp_pinned_test")
    rmtree("tmp_pinned_train")
    rmtree("tmp_pinned_valid")
