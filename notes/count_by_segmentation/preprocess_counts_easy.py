"""Preprocesses the ``counts_easy`` dataset.

Does the following:
1. Resizes the images.
2. Normalizes the images.

Produces the following plots:
1. plates.svg
2. plates_resized.svg
3. plates_normalized.svg
4. patch_variability.svg

Saves the resulting `Dataset`s.

Run ``python preprocess_counts_easy.py -h`` to see usage details.
"""

# ========================================
# Tell Python where to find cell_counting.
# ========================================
import sys, os

repo_path = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, repo_path)

# ==========================
# Import from cell_counting.
# ==========================
from cell_counting import dataset, preprocess, visualization
from models.segmentation.convnet1 import convnet1

# ===============================
# Import from the Python library.
# ===============================
import argparse, math

# ===========================
# Import from other packages.
# ===========================
from skimage import transform
import tqdm

if __name__ == "__main__":
    # ===============================
    # Process command-line arguments.
    # ===============================
    parser = argparse.ArgumentParser(description="Preprocess the counts_easy"
                                                 " dataset.")
    parser.add_argument("-outdir", type=str, required=False,
                        default="preprocess_counts_easy_output",
                        help="A path to a directory in which to save output."
                             " Will be created if nonexistent.")
    parser.add_argument("-patchsize", type=float, required=False,
                        default=43, help="The side length of the patches that"
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
    EASY_DATASET_PATH = "../../data/counts_easy"
    TQDM_PARAMS = {"desc": "load dataset", "total": 1, "unit": "dataset"}

    data_path = os.path.join(args.outdir, "counts_easy_dataset")
    loader_path = os.path.join(EASY_DATASET_PATH, "load.py")
    with tqdm.tqdm(**TQDM_PARAMS) as progress_bar:
        data = dataset.Dataset(data_path, 1)
        data.load(loader_path)
        progress_bar.update(1)

    # ==================
    # Make "plates.svg".
    # ==================
    NUM_PLATES = 5
    GRID_COLUMNS = 5
    IMAGE_SIZE = (3, 3)

    def plot_plates(filename_suffix):
        images, counts = data.get_batch(NUM_PLATES)
        filename = "plates{0:s}.svg".format(filename_suffix)
        path = os.path.join(figure_dir, filename)
        counts = counts.astype(int)
        subtitles = ["{0:d} CFU".format(counts[i]) for i in range(NUM_PLATES)]
        visualization.plot_images(images, GRID_COLUMNS, IMAGE_SIZE, "Plates",
                                  subtitles=subtitles, path=path)

    plot_plates("")

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

    # ==========================
    # Make "plates_resized.svg".
    # ==========================
    plot_plates("_resized")

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
    plot_plates("_normalized")

    # =============================
    # Make "patch_variability.svg".
    # =============================
    MIN_SIZE = 1/200
    MAX_SIZE = 1/40
    NUM_SIZES = 20
    SAMPLES = 10_000

    images, _ = data.get_all()
    min_dim, max_dim = min(images.shape[1:2]), max(images.shape[1:2])
    min_size_px = int(MIN_SIZE * min_dim)
    max_size_px = math.ceil(MAX_SIZE * max_dim)
    if min_size_px % 2 == 0:
        min_size_px -= 1
    if max_size_px % 2 == 0:
        max_size_px += 1
    min_size_px = max(min_size_px, 3)
    max_size_px = min(max_size_px, min_dim)
    sizes, var_vars = preprocess.patch_variability_curve(images, min_size_px,
                                                         max_size_px, NUM_SIZES,
                                                         SAMPLES)
    path = os.path.join(figure_dir, "patch_variability.svg")
    visualization.plot_line(sizes, var_vars, "Patch Variability Curve",
                            "patch size (px)",
                            "variance of patch variances", 4, 10, path=path)
