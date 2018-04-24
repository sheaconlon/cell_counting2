"""Preprocesses the ``more`` dataset.

Does the following:
1. Resizes the images.
2. Normalizes the images.

Saves the resulting `Dataset`s.

Run ``python preprocess_more.py -h`` to see usage details.
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
import argparse

# ===========================
# Import from other packages.
# ===========================
from skimage import transform
import tqdm

if __name__ == "__main__":
    # ===============================
    # Process command-line arguments.
    # ===============================
    PATCH_SIZE = 36

    parser = argparse.ArgumentParser(
        description="Preprocess the more dataset.")
    parser.add_argument("-outdir", type=str, required=False,
                        default="preprocess_more",
                        help="A path to a directory in which to save output."
                             " Will be created if nonexistent.")
    parser.add_argument("-patchsize", type=float, required=False,
                        default=PATCH_SIZE,
                        help="The side length of the patches that will be"
                             " extracted, in pixels.")
    args = parser.parse_args()

    # =================
    # Load the dataset.
    # =================
    TQDM_PARAMS = {"desc": "load dataset", "total": 1, "unit": "dataset"}

    data_path = os.path.join(args.outdir, "more")
    loader_path = os.path.join(repo_path, "data", "more", "load.py")
    with tqdm.tqdm(**TQDM_PARAMS) as progress_bar:
        data = dataset.Dataset(data_path, 1)
        data.load(loader_path)
        progress_bar.update(1)

    # ==================
    # Resize the images.
    # ==================
    EDGE_MODE = "reflect"
    ORDER = 3

    tqdm_params = {"desc": "resize images", "total": data.size(),
                   "unit": "image"}
    resize_factor = convnet1.ConvNet1.PATCH_SIZE / args.patchsize
    with tqdm.tqdm(**tqdm_params) as progress_bar:
        def resize_example(example):
            image, count = example
            height, width, channels = list(image.shape)
            height = round(height * resize_factor)
            width = round(width * resize_factor)
            image = transform.resize(image, (height, width), order=ORDER,
                                     mode=EDGE_MODE, clip=False)
            progress_bar.update(1)
            return [(image, count)]
        data.map(resize_example)

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
