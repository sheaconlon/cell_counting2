"""Preprocesses the ``multi`` dataset.

Does the following:
1. Resizes the images.
2. Normalizes the images.

Produces the following plots, where * is 1, 2, or 3 and + is the name of a
condition.
1. plate_*.svg
2. plate_*_resized.svg
3. plate_*_normalized.svg

Saves the resulting `Dataset`s.

Run ``python preprocess_multi.py -h`` to see usage details.
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
import numpy as np
from skimage import transform
import tqdm

if __name__ == "__main__":
    # ===============================
    # Process command-line arguments.
    # ===============================
    parser = argparse.ArgumentParser(description="Preprocess the multi"
                                                 " dataset.")
    parser.add_argument("-outdir", type=str, required=False,
                        default="preprocess_multi",
                        help="A path to a directory in which to save output."
                             " Will be created if nonexistent.")
    parser.add_argument("-patchsize", type=float, required=False,
                        default=31, help="The side length of the patches that"
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

    easy_dataset_path = os.path.join(repo_path, "data", "multi")
    data_path = os.path.join(args.outdir, "multi")
    loader_path = os.path.join(easy_dataset_path, "load.py")
    with tqdm.tqdm(**TQDM_PARAMS) as progress_bar:
        multicondition = dataset.Dataset(data_path, 1)
        multicondition.load(loader_path)
        progress_bar.update(1)

    # ===================
    # Make "plate_*.svg".
    # ===================
    NUM_PLATES = 3
    GRID_COLUMNS = 3
    IMAGE_SIZE = (4, 4)
    CONDITIONS = [
        "light_uncovered_far_noperspective",
        "nolight_uncovered_close_minorperspective",
        "light_covered_close_severeperspective"
    ]

    def plot_plates(filename_suffix):
        image_sets, counts = multicondition.get_all()
        for i in range(NUM_PLATES):
            filename = "plate_{0:d}{1:s}.svg".format(i, filename_suffix)
            path = os.path.join(figure_dir, filename)
            title = "Plate #{0:d}".format(i)
            visualization.plot_images(image_sets[i, ...], GRID_COLUMNS, IMAGE_SIZE,
                                      title, subtitles=CONDITIONS, path=path)

    plot_plates("")

    # ==================
    # Resize the images.
    # ==================
    EDGE_MODE = "reflect"
    ORDER = 3

    tqdm_params = {"desc": "resize images", "total": multicondition.size() *
                                                     len(CONDITIONS),
                   "unit": "image"}
    resize_factor = convnet1.ConvNet1.PATCH_SIZE / args.patchsize
    with tqdm.tqdm(**tqdm_params) as progress_bar:
        def resize_example(example):
            image_set, count = example
            resized_shape = list(image_set.shape)
            resized_shape[1] = round(resized_shape[1] * resize_factor)
            resized_shape[2] = round(resized_shape[2] * resize_factor)
            resized = np.empty(resized_shape)
            for i in range(image_set.shape[0]):
                resized[i, ...] = transform.resize(image_set[i, ...],
                                                   resized_shape[1:], order=ORDER,
                                                   mode=EDGE_MODE, clip=False)
                progress_bar.update(1)
            return [(resized, count)]
        multicondition.map(resize_example)

    # ===========================
    # Make "plate_*_resized.svg".
    # ===========================
    plot_plates("_resized")

    # =====================
    # Normalize the images.
    # =====================
    tqdm_params = {"desc": "normalize images",
                   "total": multicondition.size() * len(CONDITIONS),
                   "unit": "image"}
    with tqdm.tqdm(**tqdm_params) as progress_bar:
        def normalize_images(examples):
            image_sets, counts = examples
            for i in range(image_sets.shape[0]):
                image_set = image_sets[i, ...]
                image_sets[i, ...] = preprocess.divide_median_normalize(
                    image_set)
                progress_bar.update(image_sets.shape[1])
            return image_sets, counts
        multicondition.map_batch(normalize_images)

    # ==============================
    # Make "plate_*_normalized.svg".
    # ==============================
    plot_plates("_normalized")
