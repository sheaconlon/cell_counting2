"""Preprocesses the ``easy_masked`` dataset.

Does the following:
1. Resizes the images.
2. Normalizes the images.
3. Extracts patches from the images.
4. Normalizes the patches.
5. One-hot encodes the classes.
6. Splits the dataset into training and validation sets.

Produces the following plots:
1. images.svg, inside_masks.svg, edge_masks.svg, outside_masks.svg
2. images_resized.svg, inside_masks_resized.svg, edge_masks_resized.svg,
    outside_masks_resized.svg
3. images_normalized.svg
4. patches.svg
5. patches_normalized.svg

Saves the resulting `Dataset`s.

Run ``python preprocess_easy_masked.py -h`` to see usage details.
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
import argparse

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
    parser = argparse.ArgumentParser(description="Preprocess the easy_masked"
                                                 "dataset.")
    parser.add_argument("-outdir", type=str, required=False,
                        default="preprocess_easy_masked",
                        help="A path to a directory in which to save output."
                             " Will be created if nonexistent.")
    parser.add_argument("-patchsize", type=float, required=False,
                        default=43, help="The side length of the patches to"
                                         " extract, in pixels.")
    parser.add_argument("-maxpatch", type=int, required=False,
                        default=1000000, help="The maximum number of patches"
                                                " to extract from each "
                                                "example.")
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
        def transform_aspects(aspects):
            image_channels = (aspects["red"], aspects["green"],
                              aspects["blue"])
            image = np.stack(image_channels, axis=2)
            mask_channels = (aspects["inside"], aspects["edge"],
                             aspects["outside"])
            mask = np.stack(mask_channels, axis=2)
            return (image, mask)

        dataset_dir = os.path.join(args.outdir, "easy_masked")
        data = dataset.Dataset(dataset_dir, 1)
        dataset_source = os.path.join(repo_path, "data", "easy_masked", "data")
        data.initialize_from_aspects(dataset_source, transform_aspects)
        progress_bar.update(1)

    # ====================================
    # Make "images.svg" and "*_masks.svg".
    # ====================================
    def plot_images_and_masks(filename_suffix, images_only=False):
        NUM_IMAGES = 2
        GRID_COLUMNS = 2
        IMAGE_SIZE = (4, 4)

        images, masks = data.get_batch(NUM_IMAGES)
        filename = "images{0:s}.svg".format(filename_suffix)
        path = os.path.join(figure_dir, filename)
        visualization.plot_images(images, GRID_COLUMNS, IMAGE_SIZE,
                                  "Plate Images", path=path)
        if images_only:
            return
        filename = "inside_masks{0:s}.svg".format(filename_suffix)
        path = os.path.join(figure_dir, filename)
        visualization.plot_images(masks[..., 0], GRID_COLUMNS, IMAGE_SIZE,
                                  "Plate Image Inside Masks", path=path)
        filename = "edge_masks{0:s}.svg".format(filename_suffix)
        path = os.path.join(figure_dir, filename)
        visualization.plot_images(masks[..., 1], GRID_COLUMNS, IMAGE_SIZE,
                                  "Plate Image Edge Masks", path=path)
        filename = "outside_masks{0:s}.svg".format(filename_suffix)
        path = os.path.join(figure_dir, filename)
        visualization.plot_images(masks[..., 2], GRID_COLUMNS, IMAGE_SIZE,
                                  "Plate Image Outside Masks", path=path)
    plot_images_and_masks("")

    # ============================
    # Resize the images and masks.
    # ============================
    EDGE_MODE = "reflect"
    IMAGE_ORDER = 3
    MASK_ORDER = 0

    tqdm_params = {"desc": "resize images and masks", "total": data.size(),
                   "unit": "example"}
    resize_factor = convnet1.ConvNet1.PATCH_SIZE / args.patchsize
    with tqdm.tqdm(**tqdm_params) as progress_bar:
        def resize_example(example):
            image, mask = example
            target_dims = tuple(round(dim*resize_factor)
                                for dim in image.shape[:2])
            image = transform.resize(image, target_dims, order=IMAGE_ORDER,
                                     mode=EDGE_MODE, clip=False)
            mask = transform.resize(mask, target_dims, order=MASK_ORDER,
                                    mode=EDGE_MODE, clip=False)
            progress_bar.update(1)
            return [(image, mask)]
        data.map(resize_example)

    # ====================================================
    # Make "images_resized.svg" and "*_masks_resized.svg".
    # ====================================================
    plot_images_and_masks("_resized")

    # =====================
    # Normalize the images.
    # =====================
    tqdm_params = {"desc": "normalize images", "total": data.size(),
                   "unit": "example"}
    with tqdm.tqdm(**tqdm_params) as progress_bar:
        def normalize_images(examples):
            images, masks = examples
            images = preprocess.divide_median_normalize(images)
            progress_bar.update(images.shape[0])
            return (images, masks)
        data.map_batch(normalize_images)

    # =============================
    # Make "images_normalized.svg".
    # =============================
    plot_images_and_masks("_normalized", True)

    # ============================
    # Extract patches and classes.
    # ============================
    SEGMENT_PROPORTION = 0.01

    patches_dir = os.path.join(args.outdir, "patches")
    tqdm_params = {"desc": "extract patches", "total": data.size(),
                   "unit": "example"}
    with tqdm.tqdm(**tqdm_params) as progress_bar:
        def extract_patches(image, mask):
            # black is 0 and white is 255, so this gets, for each position, the
            # index of the mask with the darkest value
            class_image = np.argmin(mask, axis=2)
            yield from preprocess.extract_patches_generator(image,
                class_image, convnet1.ConvNet1.PATCH_SIZE,
                max_patches=args.maxpatch)
            progress_bar.update(1)
        segment_size = round(args.maxpatch * data.size() * SEGMENT_PROPORTION)
        new_data = data.map_generator(extract_patches, patches_dir,
                                      segment_size)
        data.delete()
        data = new_data

    # ===================
    # Make "patches.svg".
    # ===================
    def plot_patches(filename_suffix):
        NUM_PATCHES = 12
        GRID_COLUMNS = 6
        IMAGE_SIZE = (2, 2)
        CLASS_NAMES = {0: "0: inside a colony", 1: "1: on the edge of a colony",
                       2: "2: outside all colonies"}

        patches, classes = data.get_batch(NUM_PATCHES, pool_multiplier=20)
        subtitles = [CLASS_NAMES[classes[i]] for i in range(classes.shape[0])]
        path = os.path.join(figure_dir,
                            "patches{0:s}.svg".format(filename_suffix))
        visualization.plot_images(patches, GRID_COLUMNS, IMAGE_SIZE, "Patches",
                                  subtitles=subtitles, path=path)
    plot_patches("")

    # ======================
    # Normalize the patches.
    # ======================
    tqdm_params = {"desc": "normalize patches", "total": data.size(),
                   "unit": "example"}

    def normalize_patches(examples):
        patches, classes = examples
        patches = preprocess.subtract_mean_normalize(patches)
        progress_bar.update(patches.shape[0])
        return (patches, classes)

    with tqdm.tqdm(**tqdm_params) as progress_bar:
        data.map_batch(normalize_patches)

    # ==============================
    # Make "patches_normalized.svg".
    # ==============================
    plot_patches("_normalized")

    # ===========================
    # One-hot encode the classes.
    # ===========================
    NUM_CLASSES = 3

    tqdm_params = {"desc": "one-hot encode classes", "total": data.size(),
                   "unit": "example"}

    def one_hot_encode_classes(examples):
        patches, classes = examples
        one_hot_classes = np.zeros((classes.shape[0], NUM_CLASSES))
        one_hot_classes[np.arange(one_hot_classes.shape[0]), classes] = 1
        progress_bar.update(patches.shape[0])
        return (patches, one_hot_classes)

    with tqdm.tqdm(**tqdm_params) as progress_bar:
        data.map_batch(one_hot_encode_classes)

    # ====================================================
    # Split the dataset into training and validation sets.
    # ====================================================
    TEST_P = 0.1
    TRAIN_SAVE_PATH = os.path.join(args.outdir, "easy_masked_train")
    VALID_SAVE_PATH = os.path.join(args.outdir, "easy_masked_validation")

    with tqdm.tqdm(desc="split dataset", total=1) as progress_bar:
        train, test = data.split(TEST_P, TRAIN_SAVE_PATH, VALID_SAVE_PATH)
        data.delete()
        progress_bar.update(1)
