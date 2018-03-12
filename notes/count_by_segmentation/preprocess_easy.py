"""Preprocesses the ``easy`` dataset.

Does the following:
1. Resizes the images.
2. Normalizes the images.
3. Extracts patches from the images.
4. Normalizes the patches.
5. One-hot encodes the classes.

Produces the following plots:
1. images.svg
2. inside_masks.svg
3. edge_masks.svg
4. outside_masks.svg
5. patches.svg

Saves the resulting `Dataset`s as well.

Run ``python preprocess_easy.py -h`` to see usage details.
"""

# ========================================
# Tell Python where to find cell_counting.
# ========================================
import sys, os

root_relative_path = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, root_relative_path)

# ==========================
# Import from cell_counting.
# ==========================
from cell_counting import dataset, preprocess, visualization

# ===============================
# Import from the Python library.
# ===============================
import argparse

# ===========================
# Import from other packages.
# ===========================
import numpy as np
from scipy import misc
import tqdm

if __name__ == "__main__":
    # ===============================
    # Process command-line arguments.
    # ===============================
    parser = argparse.ArgumentParser(description='Preprocess the easy dataset.')
    parser.add_argument('-v', metavar='version', type=int, nargs=1,
                        help='a version number for the saved datasets',
                        default=1, required=False)
    args = parser.parse_args()
    version = args.v[0]

    # ======================
    # Make figure directory.
    # ======================
    FIGURE_BASE_PATH = "easy-{0:d}-figures".format(version)

    os.makedirs(FIGURE_BASE_PATH, exist_ok=True)

    # =================
    # Load the dataset.
    # =================
    SAVE_PATH = "easy-{0:d}-whole-images".format(version)
    EASY_PATH = "../../data/easy/data"

    with tqdm.tqdm(desc="load images/masks") as progress_bar:
        def transform_aspects(aspects):
            image_channels = (aspects["red"], aspects["green"], aspects["blue"])
            image = np.stack(image_channels, axis=2)
            mask_channels = (aspects["inside"], aspects["edge"],
                             aspects["outside"])
            mask = np.stack(mask_channels, axis=2)
            progress_bar.update(1)
            return (image, mask)
        easy = dataset.Dataset(SAVE_PATH, 1)
        easy.initialize_from_aspects(EASY_PATH, transform_aspects)

    # =======================
    # Make "originals" plots.
    # =======================
    ORIGINALS_NUM_IMAGES = 2
    ORIGINALS_GRID_COLUMNS = 2
    ORIGINALS_IMAGE_SIZE = (4, 4)
    RGB_MAX = 255

    images, masks = easy.get_batch(ORIGINALS_NUM_IMAGES)
    visualization.plot_images(images / RGB_MAX, ORIGINALS_GRID_COLUMNS,
                              ORIGINALS_IMAGE_SIZE, "Plate Images",
                              path=os.path.join(FIGURE_BASE_PATH,
                                                "images.svg"))
    visualization.plot_images(masks[..., 0] / RGB_MAX, ORIGINALS_GRID_COLUMNS,
                              ORIGINALS_IMAGE_SIZE, "Inside Masks",
                              path=os.path.join(FIGURE_BASE_PATH,
                                                "inside_masks.svg"))
    visualization.plot_images(masks[..., 1] / RGB_MAX, ORIGINALS_GRID_COLUMNS,
                              ORIGINALS_IMAGE_SIZE, "Edge Masks",
                              path=os.path.join(FIGURE_BASE_PATH,
                                                "edge_masks.svg"))
    visualization.plot_images(masks[..., 2] / RGB_MAX, ORIGINALS_GRID_COLUMNS,
                              ORIGINALS_IMAGE_SIZE, "Outside Masks",
                              path=os.path.join(FIGURE_BASE_PATH,
                                                "outside_masks.svg"))

    # ============================
    # Resize the images and masks.
    # ============================
    ACTUAL_COLONY_DIAM = 30
    TARGET_COLONY_DIAM = 61
    RESIZE_INTERP_TYPE = "bicubic"

    resize_factor = TARGET_COLONY_DIAM / ACTUAL_COLONY_DIAM
    with tqdm.tqdm(desc="resize images/masks", total=easy.size())\
            as progress_bar:
        def resize_example(example):
            image, mask = example
            target_dims = tuple(round(dim*resize_factor) for dim in
                                image.shape)
            image = misc.imresize(image, target_dims, interp=RESIZE_INTERP_TYPE)
            mask = misc.imresize(mask, target_dims, interp=RESIZE_INTERP_TYPE)
            progress_bar.update(1)
            return [(image, mask)]
        easy.map(resize_example)

    # =====================
    # Normalize the images.
    # =====================
    with tqdm.tqdm(desc="normalize images", total=easy.size()) as progress_bar:
        def normalize_examples(examples):
            images, masks = examples
            images = preprocess.divide_median_normalize(images)
            progress_bar.update(images.shape[0])
            return (images, masks)
        easy.map_batch(normalize_examples)

    # ============================
    # Extract patches and classes.
    # ============================
    MAX_PATCHES = 10_000 # 1_000_000
    SEGMENT_SIZE = 100 # 1_000
    PATCH_SAVE_PATH = "easy-{0:d}-patches".format(version)

    with tqdm.tqdm(desc="extract patches/classes from images/masks",
                   total=easy.size()) as progress_bar:
        def extract_patches(image, mask):
            class_image = np.argmin(mask, axis=2)
            yield from preprocess.extract_patches_generator(image,
                                            class_image, TARGET_COLONY_DIAM,
                                            max_patches=MAX_PATCHES)
            progress_bar.update(1)
        easy = easy.map_generator(extract_patches, PATCH_SAVE_PATH, SEGMENT_SIZE)

    # =====================
    # Make "patches" plots.
    # =====================
    PATCHES_NUM_PATCHES = 12
    PATCHES_GRID_COLUMNS = 6
    PATCHES_IMAGE_SIZE = (2, 2)
    CLASS_NAMES = {0: "inside", 1: "edge", 2: "outside"}

    images, classes = easy.get_batch(PATCHES_NUM_PATCHES)
    subtitles = [CLASS_NAMES[classes[i]] for i in range(classes.shape[0])]
    visualization.plot_images(images, PATCHES_GRID_COLUMNS, PATCHES_IMAGE_SIZE,
                              "Patches", subtitles=subtitles,
                              path=os.path.join(FIGURE_BASE_PATH,
                                                "easy_patches.svg"))

    # ======================
    # Normalize the patches.
    # ======================
    with tqdm.tqdm(desc="normalize patches", total=easy.size()) as progress_bar:
        def normalize_examples(examples):
            images, classes = examples
            images = preprocess.subtract_mean_normalize(images)
            progress_bar.update(images.shape[0])
            return (images, classes)
        easy.map_batch(normalize_examples)

    # ===========================
    # One-hot encode the classes.
    # ===========================
    NUM_CLASSES = 3

    with tqdm.tqdm(desc="one-hot encode classes", total=easy.size())\
            as progress_bar:
        def one_hot_encode_examples(examples):
            images, classes = examples
            one_hot_classes = np.zeros((classes.shape[0], NUM_CLASSES))
            one_hot_classes[np.arange(one_hot_classes.shape[0]), classes] = 1
            progress_bar.update(images.shape[0])
            return (images, one_hot_classes)
        easy.map_batch(one_hot_encode_examples)

    # ==============================================
    # Split the dataset into training and test sets.
    # ==============================================
    TEST_P = 0.1
    TRAIN_SAVE_PATH = "easy-{0:d}-patches-train".format(version)
    TEST_SAVE_PATH = "easy-{0:d}-patches-test".format(version)

    with tqdm.tqdm(desc="split dataset", total=1) as progress_bar:
        easy.split(TEST_P, TRAIN_SAVE_PATH, TEST_SAVE_PATH)
        progress_bar.update(1)
