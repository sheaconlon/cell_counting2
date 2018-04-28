"""Preprocesses the ``pinned`` dataset.

Does the following:
1. Resizes the images.
2. Normalizes the images.

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
    args = parser.parse_args()

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
    NUM_PLATES = 5
    GRID_COLUMNS = 5
    IMAGE_SIZE = (3, 3)
    EDGE_MODE = "reflect"
    ORDER = 3

    def plot_plates(filename_suffix):
        images, counts = data.get_batch(NUM_PLATES)
        filename = "{0:s}.svg".format(filename_suffix)
        path = os.path.join(figure_dir, filename)
        counts = counts.astype(int)
        subtitles = ["{0:d} CFU".format(counts[i]) for i in range(NUM_PLATES)]
        visualization.plot_images(images, GRID_COLUMNS, IMAGE_SIZE, "Groups",
                                  subtitles=subtitles, path=path)

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
    plot_plates("resized")

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
    plot_plates("normalized")
