"""Preprocesses the ``pinned`` dataset.

Does the following:
1. Augments the images.
2. Resizes the images.
3. Normalizes the images.
4. Discretizes the counts.
5. HOG-featurizes the images.
6. Splits the dataset.

Produces the following plots:
1. plates_resized.svg
2. plates_normalized.svg
3. plates_discretized.svg
4. plates_hog_featurized.svg

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
from tqdm import tqdm
from skimage import feature, color
from imgaug import augmenters as iaa

NUM_PLOT_PLATES = 5
BINS = (5, 10, 20)


def plot_plates(images, outputs, base_filename, discretized=False):
    padded_bins = (0,) + BINS + ("Inf",)
    filename = "{0:s}.svg".format(base_filename)
    path = os.path.join(figure_dir, filename)
    outputs = outputs.astype(int)
    if discretized:
        subtitles = []
        for i in range(NUM_PLOT_PLATES):
            if outputs[i] == 0:
                subtitles.append("OUTLIER")
            else:
                first = str(padded_bins[outputs[i] - 1])
                second = str(padded_bins[outputs[i]])
                subtitles.append("{0:s}-{1:s}".format(first, second))
    else:
        subtitles = ["{0:d} CFU".format(outputs[i])
                     for i in range(NUM_PLOT_PLATES)]
    visualization.plot_images(images, GRID_COLUMNS, IMAGE_SIZE, "Well Images",
                              subtitles=subtitles, path=path)

def make_duplicator(factor):
    def duplicate_batch(batch):
        inputs, outputs = batch
        inputs = np.concatenate([inputs] * factor, axis=0)
        outputs = np.concatenate([outputs] * factor, axis=0)
        return inputs, outputs
    return duplicate_batch


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
    parser.add_argument("-numaugs", type=int, required=False, default=2,
                        help = "The number of augmented versions to produce per"
                               " example.")
    parser.add_argument("-patchsize", type=float, required=False,
                        default=40, help="The side length of the patches that"
                                         " will be extracted, in pixels.")
    parser.add_argument("-validp", type=float, required=False,
                        default=0.2, help="The proportion of the examples to"
                                          " put in the validation split.")
    parser.add_argument("-testp", type=float, required=False,
                        default=0.2, help="The proportion of the examples to"
                                          " put in the test split.")
    parser.add_argument("-sizefactor", type=float, required=False,
                        default=1, help="A factor to scale the well images by.")
    args = parser.parse_args()

    assert 0 < args.validp < 1, "Argument 'validp' must be in (0, 1)."
    assert 0 < args.testp < 1, "Argument 'testp' must be in (0, 1)."
    assert args.sizefactor > 0, "argument 'sizefactor' must be positive"

    # ======================
    # Make figure directory.
    # ======================
    figure_dir = os.path.join(args.outdir, "figures")
    os.makedirs(figure_dir, exist_ok=True)

    # =================
    # Load the dataset.
    # =================
    TQDM_PARAMS = {"desc": "load dataset", "total": 1, "unit": "dataset"}

    with tqdm(**TQDM_PARAMS) as progress_bar:
        path = os.path.join(args.outdir, "pinned")
        data = dataset.Dataset(path, 1)
        path = os.path.join(repo_path, "data", "pinned", "load.py")
        data.load(path)
        progress_bar.update(1)

    # # ===================
    # # Augment the images.
    # # ===================
    # OFTEN, SOMETIMES, RARELY = 0.25, 0.05, 0.02
    #
    # inputs = iaa.Sequential([
    #     iaa.Fliplr(OFTEN),
    #     iaa.Flipud(OFTEN),
    #     iaa.Sometimes(SOMETIMES, iaa.PerspectiveTransform((0, 0.05))),
    #     iaa.Invert(RARELY, per_channel=True),
    #     iaa.Sometimes(SOMETIMES, iaa.Add((-45, 45), per_channel=True)),
    #     iaa.Sometimes(RARELY,
    #                   iaa.AddToHueAndSaturation(value=(-15, 15),
    #                                             from_colorspace="RGB")),
    #     iaa.Sometimes(SOMETIMES, iaa.GaussianBlur(sigma=(0, 1))),
    #     iaa.Sometimes(RARELY,
    #                   iaa.Sharpen(alpha=(0, 0.25), lightness=(0.9, 1.1))),
    #     iaa.Sometimes(SOMETIMES,
    #                   iaa.AdditiveGaussianNoise(scale=(0, 0.02 * 255))),
    #     iaa.SaltAndPepper(SOMETIMES),
    #     iaa.Sometimes(SOMETIMES, iaa.ContrastNormalization((0.5, 1.5))),
    #     iaa.Sometimes(SOMETIMES, iaa.Grayscale(alpha=(0.0, 1.0)))
    # ])
    #
    # with tqdm(desc="augment images", total=1, unit="dataset") as prog:
    #     data.map_batch(make_duplicator(args.numaugs))
    #     data.augment(input_augmenter=inputs)
    #     prog.update(1)

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
    with tqdm(**tqdm_params) as progress_bar:
        def check_size(example):
            global max_size
            image, count = example
            max_size = max(max_size, max(image.shape[0], image.shape[1]))
            progress_bar.update(1)
            return [(image, count)]
        data.map(check_size)

    tqdm_params = {"desc": "resize images", "total": data.size(),
                   "unit": "image"}
    resize_factor = convnet1.ConvNet1.PATCH_SIZE / args.patchsize \
                                                            * args.sizefactor
    target_size = (round(max_size * resize_factor),
                   round(max_size * resize_factor))
    with tqdm(**tqdm_params) as progress_bar:
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
    images, counts = data.get_batch(NUM_PLOT_PLATES)
    plot_plates(images, counts, "plates_resized")

    # =====================
    # Normalize the images.
    # =====================
    tqdm_params = {"desc": "normalize images", "total": data.size(),
                   "unit": "image"}
    with tqdm(**tqdm_params) as progress_bar:
        def normalize_images(examples):
            images, counts = examples
            images = preprocess.divide_median_normalize(images)
            progress_bar.update(1)
            return images, counts
        data.map_batch(normalize_images)

    # =============================
    # Make "plates_normalized.svg".
    # =============================
    images, counts = data.get_batch(NUM_PLOT_PLATES)
    plot_plates(images, counts, "plates_normalized")

    # ======================
    # Discretize the counts.
    # ======================
    with tqdm(desc="Discretizing counts", unit="example",
              total=data.size()) as prog:
        def discretize(batch):
            images, counts = batch
            for i in range(counts.shape[0]):
                if counts[i] == -1:
                    counts[i] = 0
                else:
                    for bin, bound in enumerate(BINS + (float("inf"),)):
                        if counts[i] <= bound:
                            counts[i] = bin + 1
                            break
            prog.update(images.shape[0])
            return images, counts
        data.map_batch(discretize)

    # ==============================
    # Make "plates_discretized.svg".
    # ==============================
    images, counts = data.get_batch(NUM_PLOT_PLATES)
    plot_plates(images, counts, "plates_discretized", discretized=True)

    # =========================
    # HOG-featurize the images.
    # =========================
    hog_vizs = []
    hog_viz_counts = []
    with tqdm(desc="HOG-featurizing images", unit="example",
              total=data.size()) as prog:
        def hog(example):
            im, count = example
            make_viz = len(hog_vizs) < NUM_PLOT_PLATES
            result = feature.hog(im, orientations=9, pixels_per_cell=(8, 8),
                                 cells_per_block=(3, 3), block_norm="L2-Hys",
                                 feature_vector=True, visualize=make_viz)
            if make_viz:
                hog_im, viz = result
                hog_vizs.append(viz)
                hog_viz_counts.append(count)
            else:
                hog_im = result
            prog.update(1)
            return [(hog_im, count)]
        data.map(hog)

    # =================================
    # Make "plates_hog_featurized.svg".
    # =================================
    images = np.stack(hog_vizs, axis=0)
    counts = np.stack(hog_viz_counts, axis=0)
    plot_plates(images, counts, "plates_hog_featurized", discretized=True)

    # ==================
    # Split the dataset.
    # ==================
    with tqdm(desc="splitting dataset", total=2, unit="splits") as prog:
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
