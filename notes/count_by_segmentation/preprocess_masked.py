"""Preprocesses the ``easy_masked`` and ``more_masked`` datasets.

Does the following:
1. Resizes the images.
2. Normalizes the images.
3. Extracts patches from the images.
4. Normalizes the patches.
5. One-hot encodes the classes.
6. Splits the dataset into training and validation sets.

Produces the following plots, where * is one of "easy" or "more".
1. */images.svg, */inside_masks.svg, */edge_masks.svg, */outside_masks.svg
2. */images_resized.svg, */inside_masks_resized.svg, */edge_masks_resized.svg,
    */outside_masks_resized.svg
3. */images_normalized.svg
4. patches.svg
5. patches_normalized.svg

Saves the resulting `Dataset`s.

Run ``python preprocess_masked.py -h`` to see usage details.
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
    parser = argparse.ArgumentParser(description="Preprocess the masked"
                                                 " dataset.")
    parser.add_argument("-outdir", type=str, required=False,
                        default="preprocess_masked",
                        help="A path to a directory in which to save output."
                             " Will be created if nonexistent.")
    parser.add_argument("-easypatchsize", type=float, required=False,
                        default=43, help="The side length of the patches to"
                                         " extract from easy_masked, in"
                                         " pixels.")
    parser.add_argument("-morepatchsize", type=float, required=False,
                        default=59, help="The side length of the patches to"
                                        " extract from more_masked, in"
                                        " pixels.")
    parser.add_argument("-maxpatch", type=int, required=False,
                        default=1000000, help="The maximum number of patches"
                                                " to extract from each "
                                                "example.")
    args = parser.parse_args()

    # ========================
    # Make figure directories.
    # ========================
    figure_dir = os.path.join(args.outdir, "figures")
    path = os.path.join(figure_dir, "easy")
    os.makedirs(path, exist_ok=True)
    path = os.path.join(figure_dir, "more")
    os.makedirs(path, exist_ok=True)

    # ==================
    # Load the datasets.
    # ==================
    TQDM_PARAMS = {"desc": "load datasets", "total": 2, "unit": "dataset"}

    with tqdm.tqdm(**TQDM_PARAMS) as progress_bar:
        def transform_aspects(aspects):
            image_channels = (aspects["red"], aspects["green"],
                              aspects["blue"])
            image = np.stack(image_channels, axis=2)
            mask_channels = (aspects["inside"], aspects["edge"],
                             aspects["outside"])
            mask = np.stack(mask_channels, axis=2)
            return (image, mask)

        path = os.path.join(args.outdir, "easy_masked")
        easy_masked = dataset.Dataset(path, 1)
        path = os.path.join(repo_path, "data", "easy_masked", "data")
        easy_masked.initialize_from_aspects(path, transform_aspects)
        progress_bar.update(1)

        path = os.path.join(args.outdir, "more_masked")
        more_masked = dataset.Dataset(path, 1)
        path = os.path.join(repo_path, "data", "more_masked", "data")
        more_masked.initialize_from_aspects(path, transform_aspects)
        progress_bar.update(1)

    # ====================================
    # Make "images.svg" and "*_masks.svg".
    # ====================================
    def plot_images_and_masks(dataset, dataset_name, filename_suffix,
                                images_only=False):
        IMAGE_SIZE = (4, 4)

        images, masks = dataset.get_all()

        filename = "images{0:s}.svg".format(filename_suffix)
        path = os.path.join(figure_dir, dataset_name, filename)
        visualization.plot_images(images, images.shape[0], IMAGE_SIZE,
                                  "Plate Images", path=path)
        if images_only:
            return
        filename = "inside_masks{0:s}.svg".format(filename_suffix)
        path = os.path.join(figure_dir, dataset_name, filename)
        visualization.plot_images(masks[..., 0], images.shape[0], IMAGE_SIZE,
                                  "Plate Image Inside Masks", path=path)
        filename = "edge_masks{0:s}.svg".format(filename_suffix)
        path = os.path.join(figure_dir, dataset_name, filename)
        visualization.plot_images(masks[..., 1], images.shape[0], IMAGE_SIZE,
                                  "Plate Image Edge Masks", path=path)
        filename = "outside_masks{0:s}.svg".format(filename_suffix)
        path = os.path.join(figure_dir, dataset_name, filename)
        visualization.plot_images(masks[..., 2], images.shape[0], IMAGE_SIZE,
                                  "Plate Image Outside Masks", path=path)
    plot_images_and_masks(easy_masked, "easy", "")
    plot_images_and_masks(more_masked, "more", "")

    # ============================
    # Resize the images and masks.
    # ============================
    EDGE_MODE = "reflect"
    IMAGE_ORDER = 3
    MASK_ORDER = 0

    tqdm_params = {"desc": "resize images and masks",
                    "total": easy_masked.size() + more_masked.size(),
                    "unit": "example"}
    
    easy_factor = convnet1.ConvNet1.PATCH_SIZE / args.easypatchsize
    more_factor = convnet1.ConvNet1.PATCH_SIZE / args.morepatchsize

    with tqdm.tqdm(**tqdm_params) as progress_bar:
        def make_resizer(factor):
            def resizer(example):
                image, mask = example
                target_dims = tuple(round(dim*factor)
                                    for dim in image.shape[:2])
                image = transform.resize(image, target_dims, order=IMAGE_ORDER,
                                         mode=EDGE_MODE, clip=False)
                mask = transform.resize(mask, target_dims, order=MASK_ORDER,
                                        mode=EDGE_MODE, clip=False)
                progress_bar.update(1)
                return [(image, mask)]
            return resizer
        easy_masked.map(make_resizer(easy_factor))
        more_masked.map(make_resizer(more_factor))

    # ====================================================
    # Make "images_resized.svg" and "*_masks_resized.svg".
    # ====================================================
    plot_images_and_masks(easy_masked, "easy", "_resized")
    plot_images_and_masks(more_masked, "more", "_resized")

    # =====================
    # Normalize the images.
    # =====================
    tqdm_params = {"desc": "normalize images",
                    "total": easy_masked.size() + more_masked.size(),
                    "unit": "example"}
    with tqdm.tqdm(**tqdm_params) as progress_bar:
        def normalize_images(examples):
            images, masks = examples
            images = preprocess.divide_median_normalize(images)
            progress_bar.update(images.shape[0])
            return (images, masks)
        easy_masked.map_batch(normalize_images)
        more_masked.map_batch(normalize_images)

    # =============================
    # Make "images_normalized.svg".
    # =============================
    plot_images_and_masks(easy_masked, "easy", "_normalized", True)
    plot_images_and_masks(more_masked, "more", "_normalized", True)

    # ============================
    # Extract patches and classes.
    # ============================
    SEGMENT_PROPORTION = 0.01

    tqdm_params = {"desc": "extract patches",
                    "total": easy_masked.size() + more_masked.size(),
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

        segment_size = round(args.maxpatch * easy_masked.size() \
                            * SEGMENT_PROPORTION)
        path = os.path.join(args.outdir, "easy_masked_patched")
        easy_masked = easy_masked.map_generator(extract_patches, path,
                                        segment_size)

        segment_size = round(args.maxpatch * more_masked.size() \
                            * SEGMENT_PROPORTION)
        path = os.path.join(args.outdir, "more_masked_patched")
        more_masked = more_masked.map_generator(extract_patches, path,
                                        segment_size)

    # ===================
    # Merge the datasets.
    # ===================
    path = os.path.join(args.outdir, "masked")
    size = easy_masked.size() + more_masked.size()
    segment_size = round(size * SEGMENT_PROPORTION)
    masked = dataset.Dataset(path, segment_size)
    masked.add(easy_masked)
    masked.add(more_masked)

    # ===================
    # Make "patches.svg".
    # ===================
    def plot_patches(filename_suffix):
        NUM_PATCHES = 12
        GRID_COLUMNS = 6
        IMAGE_SIZE = (2, 2)
        CLASS_NAMES = {0: "0: inside a colony", 1: "1: on the edge of a colony",
                       2: "2: outside all colonies"}

        patches, classes = masked.get_batch(NUM_PATCHES, pool_multiplier=20)
        subtitles = [CLASS_NAMES[classes[i]] for i in range(classes.shape[0])]
        path = os.path.join(figure_dir,
                            "patches{0:s}.svg".format(filename_suffix))
        visualization.plot_images(patches, GRID_COLUMNS, IMAGE_SIZE, "Patches",
                                  subtitles=subtitles, path=path)
    plot_patches("")

    # ======================
    # Normalize the patches.
    # ======================
    tqdm_params = {"desc": "normalize patches", "total": masked.size(),
                   "unit": "example"}

    def normalize_patches(examples):
        patches, classes = examples
        patches = preprocess.subtract_mean_normalize(patches)
        progress_bar.update(patches.shape[0])
        return (patches, classes)

    with tqdm.tqdm(**tqdm_params) as progress_bar:
        masked.map_batch(normalize_patches)

    # ==============================
    # Make "patches_normalized.svg".
    # ==============================
    plot_patches("_normalized")

    # ===========================
    # One-hot encode the classes.
    # ===========================
    NUM_CLASSES = 3

    tqdm_params = {"desc": "one-hot encode classes", "total": masked.size(),
                   "unit": "example"}

    def one_hot_encode_classes(examples):
        patches, classes = examples
        one_hot_classes = np.zeros((classes.shape[0], NUM_CLASSES))
        one_hot_classes[np.arange(one_hot_classes.shape[0]), classes] = 1
        progress_bar.update(patches.shape[0])
        return (patches, one_hot_classes)

    with tqdm.tqdm(**tqdm_params) as progress_bar:
        masked.map_batch(one_hot_encode_classes)

    # ====================================================
    # Split the dataset into training and validation sets.
    # ====================================================
    TEST_P = 0.1
    TRAIN_SAVE_PATH = os.path.join(args.outdir, "masked_train")
    VALID_SAVE_PATH = os.path.join(args.outdir, "masked_validation")

    with tqdm.tqdm(desc="split dataset", total=1) as progress_bar:
        train, test = masked.split(TEST_P, TRAIN_SAVE_PATH, VALID_SAVE_PATH)
        masked.delete()
        progress_bar.update(1)
