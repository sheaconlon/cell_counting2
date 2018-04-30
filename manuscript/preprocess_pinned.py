"""Preprocesses the ``pinned`` dataset.

Does the following:
1. Resizes the images.
2. Normalizes the images.
3. Splits the dataset.

Produces the following plots:
2. resized.svg
3. normalized.svg

Saves the resulting `Dataset`s.

Run ``python preprocess_pinned.py -h`` to see usage details.
"""

# ========================================
# Tell Python where to find cell_counting.
# ========================================
import sys, os

repo_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, repo_path)

# ==========================
# Import from cell_counting.
# ==========================
from cell_counting import dataset, preprocess, visualization
from cell_counting.models.segmentation import convnet1

# ===============================
# Import from the Python library.
# ===============================
import argparse, math

# ===========================
# Import from other packages.
# ===========================
from skimage import transform
import numpy as np
import tqdm


def plot_plates(base_filename):
    NUM_PLATES = 5

    images, counts = data.get_batch(NUM_PLATES)
    filename = "{0:s}.svg".format(base_filename)
    path = os.path.join(figure_dir, filename)
    counts = counts.astype(int)
    subtitles = ["{0:d} CFU".format(counts[i]) for i in range(NUM_PLATES)]
    visualization.plot_images(images, GRID_COLUMNS, IMAGE_SIZE, "Groups",
                              subtitles=subtitles, path=path)


if __name__ == "__main__":
    # ===============================
    # Process command-line arguments.
    # ===============================
    parser = argparse.ArgumentParser(description="Preprocess the pinned "
                                                 "dataset.")
    parser.add_argument("-outdir", type=str, required=False,
                        default="preprocess_pinned",
                        help="A path to a directory in which to save output."
                             " Will be created if nonexistent.")
    parser.add_argument("-patchsize", type=float, required=False,
                        default=40, help="The side length of the patches that"
                                         " will be extracted, in pixels.")
    parser.add_argument("-validp", type=float, required=False,
                        default=0.2, help="The proportion of the examples to"
                                          " put in the validation split.")
    parser.add_argument("-testp", type=float, required=False,
                        default=0.2, help="The proportion of the examples to"
                                          " put in the test split.")
    args = parser.parse_args()

    assert 0 < args.validp < 1, "Argument 'validp' must be in (0, 1)."
    assert 0 < args.testp < 1, "Argument 'testp' must be in (0, 1)."

    # ======================
    # Make figure directory.
    # ======================
    figure_dir = os.path.join(args.outdir, "figures")
    os.makedirs(figure_dir, exist_ok=True)

    # =================
    # Load the dataset.
    # =================
    TQDM_PARAMS = {"desc": "load dataset", "total": 1, "unit": "dataset"}

    with tqdm.tqdm(**TQDM_PARAMS) as progress_bar:
        path = os.path.join(args.outdir, "pinned")
        data = dataset.Dataset(path, 1)
        path = os.path.join(repo_path, "data", "pinned", "load.py")
        data.load(path)
        progress_bar.update(1)

    # ==================
    # Resize the images.
    # ==================
    GRID_COLUMNS = 5
    IMAGE_SIZE = (3, 3)
    EDGE_MODE = "reflect"
    ORDER = 3

    tqdm_params = {"desc": "determine max size", "total": data.size(),
                   "unit": "image"}
    max_size = float("-inf")
    with tqdm.tqdm(**tqdm_params) as progress_bar:
        def check_size(example):
            global max_size
            image, count = example
            max_size = max(max_size, max(image.shape[0], image.shape[1]))
            progress_bar.update(1)
            return [(image, count)]
        data.map(check_size)

    tqdm_params = {"desc": "resize images", "total": data.size(),
                   "unit": "image"}
    resize_factor = convnet1.ConvNet1.PATCH_SIZE / args.patchsize
    target_size = (round(max_size * resize_factor),
                   round(max_size * resize_factor))
    with tqdm.tqdm(**tqdm_params) as progress_bar:
        def resize_example(example):
            image, count = example
            image = np.ascontiguousarray(image)
            image = transform.resize(image, target_size, order=ORDER,
                                     mode=EDGE_MODE, clip=False)
            progress_bar.update(1)
            return [(image, count)]
        data.map(resize_example)

    # ==========================
    # Make "plates_resized.svg".
    # ==========================
    plot_plates("plates_resized")

    # =====================
    # Normalize the images.
    # =====================
    tqdm_params = {"desc": "normalize images", "total": data.size(),
                   "unit": "image"}
    with tqdm.tqdm(**tqdm_params) as progress_bar:
        def normalize_images(examples):
            images, counts = examples
            images = preprocess.divide_median_normalize(images)
            progress_bar.update(1)
            return images, counts
        data.map_batch(normalize_images)

    # =============================
    # Make "plates_normalized.svg".
    # =============================
    plot_plates("plates_normalized")

    # ==================
    # Split the dataset.
    # ==================
    with tqdm.tqdm(desc="splitting dataset", total=2, unit="splits") as prog:
        path_train_valid = os.path.join(args.outdir, "pinned_train_valid")
        path_test = os.path.join(args.outdir, "pinned_test")
        train_valid, test = data.split(args.testp, path_train_valid, path_test)
        prog.update(1)

        validp = args.validp / (1 - args.testp)
        path_train = os.path.join(args.outdir, "pinned_train")
        path_valid = os.path.join(args.outdir, "pinned_valid")
        train, valid = data.split(validp, path_train, path_valid)
        prog.update(1)

    train_valid.delete()
