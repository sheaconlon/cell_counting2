"""Preprocesses the ``easy_masked`` and ``more_masked`` datasets.

Does the following:
1. Resizes the images.
2. Normalizes the images.
3. Extracts patches from the images.
4. Augments the patches.
5. Normalizes the patches.
6. One-hot encodes the classes.
7. Splits the dataset into training and validation sets.

Produces the following plots, where * is one of "easy" or "more".
1. */images.svg, */inside_masks.svg, */edge_masks.svg, */outside_masks.svg
2. */images_resized.svg, */inside_masks_resized.svg, */edge_masks_resized.svg,
    */outside_masks_resized.svg
3. */images_normalized.svg
4. patches.svg
5. patches_augmented.svg
6. patches_normalized.svg

Saves the resulting `Dataset`s.

Run ``python preprocess_masked.py -h`` to see usage details.
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
import random

# ===========================
# Import from other packages.
# ===========================
import numpy as np
from skimage import transform
import tqdm
from imgaug import augmenters as iaa

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
    SCALE_MIN = 0.2
    SCALE_MODE = 0.8
    SCALE_MAX = 1.5
    NUM_SCALES_PER_IMAGE = 2  # FIXME
    
    easy_factor = convnet1.ConvNet1.PATCH_SIZE / args.easypatchsize
    more_factor = convnet1.ConvNet1.PATCH_SIZE / args.morepatchsize

    def make_resizer(factor):
        def resizer(image, mask):
            target_dims = tuple(round(dim * factor) for dim in image.shape[:2])
            image = transform.resize(image, target_dims, order=IMAGE_ORDER,
                                     mode=EDGE_MODE, clip=False)
            mask = transform.resize(mask, target_dims, order=MASK_ORDER,
                                    mode=EDGE_MODE, clip=False)
            example = (image, mask)
            prog.update(1)
            yield example
        return resizer

    def resize_rescale(dataset, name, factor):
        datasets = []
        for i in range(NUM_SCALES_PER_IMAGE):
            path = os.path.join(args.outdir, name + "_scaled_" + str(i))
            scale = random.triangular(SCALE_MIN, SCALE_MAX, SCALE_MODE)
            resized_rescaled = dataset.map_generator(
                make_resizer(factor*scale), path, 1)
            datasets.append(resized_rescaled)
        return datasets

    resizes = (easy_masked.size() + more_masked.size()) * NUM_SCALES_PER_IMAGE
    with tqdm.tqdm(desc="resize images and masks", unit="example-scale",
                   total=resizes) as prog:
        easy_scales = resize_rescale(easy_masked, "easy_masked", easy_factor)
        more_scales = resize_rescale(more_masked, "more_masked", more_factor)

    # ====================================================
    # Make "images_resized.svg" and "*_masks_resized.svg".
    # ====================================================
    for i in range(NUM_SCALES_PER_IMAGE):
        plot_images_and_masks(easy_scales[i], "easy", "_resized_" + str(i))
    for i in range(NUM_SCALES_PER_IMAGE):
        plot_images_and_masks(more_scales[i], "more", "_resized_" + str(i))

    # =====================
    # Normalize the images.
    # =====================
    tqdm_params = {"desc": "normalize images",
                    "total": resizes,
                    "unit": "example"}
    with tqdm.tqdm(**tqdm_params) as progress_bar:
        def normalize_images(examples):
            images, masks = examples
            images = preprocess.divide_median_normalize(images)
            progress_bar.update(images.shape[0])
            return (images, masks)
        for i in range(NUM_SCALES_PER_IMAGE):
            easy_scales[i].map_batch(normalize_images)
        for i in range(NUM_SCALES_PER_IMAGE):
            more_scales[i].map_batch(normalize_images)

    # =============================
    # Make "images_normalized.svg".
    # =============================
    for i in range(NUM_SCALES_PER_IMAGE):
        plot_images_and_masks(easy_scales[i], "easy", "_normalized_" + str(i))
    for i in range(NUM_SCALES_PER_IMAGE):
        plot_images_and_masks(more_scales[i], "more", "_normalized_" + str(i))

    # ============================
    # Extract patches and classes.
    # ============================
    SEGMENT_PROPORTION = 0.01
    MAX_PATCH_FACTOR_PER_SCALE = 1/NUM_SCALES_PER_IMAGE

    with tqdm.tqdm(desc="extract patches", total=resizes,
                   unit="example-scale") as prog:
        max_patch = int(args.maxpatch * MAX_PATCH_FACTOR_PER_SCALE)

        def extractor(image, mask):
            # black is 0 and white is 255, so this gets, for each position, the
            # index of the mask with the darkest value
            class_image = np.argmin(mask, axis=2)
            yield from preprocess.extract_patches_generator(image,
                class_image, convnet1.ConvNet1.PATCH_SIZE,
                max_patches=max_patch)
            prog.update(1)

        def patchify(dataset, name):
            segment_size = round(max_patch * dataset.size()
                                 * SEGMENT_PROPORTION)
            path = os.path.join(args.outdir, name + "_patched")
            patched = dataset.map_generator(extractor, path, segment_size)
            return patched
        for i in range(NUM_SCALES_PER_IMAGE):
            easy_scales[i] = patchify(easy_scales[i], "easy_" + str(i))
        for i in range(NUM_SCALES_PER_IMAGE):
            more_scales[i] = patchify(more_scales[i], "more_" + str(i))

    # ===================
    # Merge the datasets.
    # ===================
    path = os.path.join(args.outdir, "masked")
    size = 0
    for i in range(NUM_SCALES_PER_IMAGE):
        size += easy_scales[i].size()
    for i in range(NUM_SCALES_PER_IMAGE):
        size += more_scales[i].size()
    segment_size = round(size * SEGMENT_PROPORTION)
    masked = dataset.Dataset(path, segment_size)
    for i in range(NUM_SCALES_PER_IMAGE):
        masked.add(easy_scales[i])
    for i in range(NUM_SCALES_PER_IMAGE):
        masked.add(more_scales[i])

    # ===================
    # Make "patches.svg".
    # ===================
    def plot_patches(filename_suffix):
        NUM_PATCHES = 24
        GRID_COLUMNS = 6
        IMAGE_SIZE = (2, 2)
        CLASS_NAMES = {0: "0: inside a colony", 1: "1: on the edge of a colony",
                       2: "2: outside all colonies"}

        patches, classes = masked._load_segment(0)
        patches, classes = patches[:NUM_PATCHES, ...], classes[:NUM_PATCHES]
        subtitles = [CLASS_NAMES[classes[i]] for i in range(classes.shape[0])]
        path = os.path.join(figure_dir,
                            "patches{0:s}.svg".format(filename_suffix))
        visualization.plot_images(patches, GRID_COLUMNS, IMAGE_SIZE, "Patches",
                                  subtitles=subtitles, path=path)
    plot_patches("")

    # ====================
    # Augment the patches.
    # ====================
    ALWAYS, OFTEN, SOMETIMES, RARELY = 1, 1/2, 1/10, 1/100
    seq = iaa.Sequential([
        iaa.Fliplr(OFTEN),
        # iaa.Flipud(ALWAYS),
        # iaa.Invert(RARELY, per_channel=True),
        # iaa.Sometimes(SOMETIMES, iaa.Add((-45, 45), per_channel=True)),
        # iaa.Sometimes(RARELY, iaa.AddToHueAndSaturation(value=(-15, 15),
        #                                         from_colorspace="RGB")),
        # iaa.Sometimes(SOMETIMES, iaa.GaussianBlur(sigma=(0, 1))),
        # iaa.Sometimes(RARELY, iaa.Sharpen(alpha=(0, 0.25),
        #                                  lightness=(0.9, 1.1))),
        # iaa.Sometimes(SOMETIMES, iaa.AdditiveGaussianNoise(
        #                                            scale=(0, 0.02 * 255))),
        # iaa.SaltAndPepper(SOMETIMES),
        # iaa.Sometimes(SOMETIMES, iaa.ContrastNormalization((0.5, 1.5))),
        # iaa.Sometimes(SOMETIMES, iaa.Grayscale(alpha=(0.0, 1.0))),
        # iaa.Sometimes(SOMETIMES, iaa.PerspectiveTransform((0, 0.05)))
    ])
    with tqdm.tqdm(desc="augment patches", total=1, unit="dataset") as prog:
        masked.augment(seq)
        prog.update(1)

    plot_patches("_augmented")

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
