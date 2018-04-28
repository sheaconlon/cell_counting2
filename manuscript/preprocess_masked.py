"""Preprocesses the ``easy_masked`` and ``more_masked`` datasets.

Does the following:
1. Augments the images (each to multiple versions).
2. Resizes the images (each to mutiple scales).
3. Normalizes the images.
4. Extracts patches from the images.
5. Normalizes the patches.
6. One-hot encodes the classes.
7. Splits the dataset into training and validation sets.

Produces the following plots, where * is one of "easy" or "more", + is an
integer, and & is one of "image", "inside", "edge", or "outside".
1. original/*_+_&.png
2. augmented/*_+_&.png
3. resized/*_+_&.png
4. normalized/*_+_&.png
5. patched_original.png
5. patched_normalized.png
6. patched_one_hot_encoded.svg

Saves the resulting `Dataset`s.

Run ``python preprocess_masked.py -h`` to see usage details.
"""

# Tell Python where to find cell_counting.
# ========================================
import sys
import os
repo_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, repo_path)

# Import from cell_counting.
# ==========================
from cell_counting import dataset, preprocess, visualization
from cell_counting.models.segmentation import convnet1

# Import from the Python library.
# ===============================
import argparse
import random
import shutil

# Import from other packages.
# ===========================
import numpy as np
from skimage import transform
import tqdm
from imgaug import augmenters as iaa
import imageio

if __name__ == "__main__":
    # Process command-line arguments.
    # ===============================
    parser = argparse.ArgumentParser(
        description="Preprocess the masked dataset.")
    parser.add_argument("-outdir", type=str, required=False,
        help="The directory in which to save output. Will be created if"
             " nonexistent.", default="preprocess_masked")
    parser.add_argument("-easypatchsize", type=float, required=False,
        help="The side length of the patches to extract from easy_masked, in"
             " pixels. Must be positive.", default=34)
    parser.add_argument("-morepatchsize", type=float, required=False,
        help="The side length of the patches to extract from more_masked, in"
             " pixels. Must be positive.", default=47)
    parser.add_argument("-maxpatches", type=int, required=False,
        help="The maximum number of patches to produce.",
        default=1000000)
    parser.add_argument("-numaugs", type=int, required=False, default=10,
        help="The number of augmented versions to produce per example.")
    parser.add_argument("-numscales", type=int, required=False, default=10,
        help="The number of scaled versions to produce per augmented example.")
    args = parser.parse_args()
    assert args.easypatchsize > 0, "EASYPATCHSIZE must be positive"
    assert args.morepatchsize > 0, "MOREPATCHSIZE must be positive"

    # Define some helper functions.
    # =============================
    figure_path = os.path.join(args.outdir, "figures")
    shutil.rmtree(args.outdir, ignore_errors=True)

    def transform_aspects(aspects):
        image_channels = (aspects["red"], aspects["green"],
                          aspects["blue"])
        image = np.stack(image_channels, axis=2)
        mask_channels = (aspects["inside"], aspects["edge"],
                         aspects["outside"])
        mask = np.stack(mask_channels, axis=2)
        return (image, mask)

    def save_image(path, image):
        RGB_MAX = 255

        image_min = np.amin(image)
        if image_min < 0:
            image += -1 * image_min
        image_max = np.amax(image)
        if image_max > RGB_MAX:
            image = image * (RGB_MAX / image_max)
        imageio.imsave(path, image.astype(np.uint8))

    def plot_images(dataset, subdir, name, factor=1, start_num=0, limit=None):
        base = os.path.join(figure_path, subdir)
        os.makedirs(base, exist_ok=True)
        images, masks = dataset.get_all()
        images, masks = images[:limit, ...], masks[:limit, ...]
        images, masks = images*factor, masks
        for i in range(images.shape[0]):
            filename = "{0:s}_{1:d}_{2:s}.png".format(name, start_num + i,
                                                      "image")
            save_image(os.path.join(base, filename), images[i, ...])

            for j, mask in enumerate(("inside", "edge", "outside")):
                filename = "{0:s}_{1:d}_{2:s}.png".format(name, start_num + i,
                                                          mask)
                save_image(os.path.join(base, filename), masks[i, ..., j])

    def make_duplicator(factor):
        def duplicate_batch(batch):
            inputs, outputs = batch
            inputs = np.concatenate([inputs] * factor, axis=0)
            outputs = np.concatenate([outputs] * factor, axis=0)
            return inputs, outputs
        return duplicate_batch

    def plot_patches(dataset, name, factor=1):
        NUM_PATCHES = 100
        GRID_COLUMNS = 10
        IMAGE_SIZE = (2, 2)
        CLASS_NAMES = {0: "0: inside", 1: "1: on edge", 2: "2: outside"}

        patches, classes = dataset.get_batch(NUM_PATCHES, pool_multiplier=20)
        patches, classes = patches[:NUM_PATCHES, ...], classes[:NUM_PATCHES]
        patches *= factor
        subtitles = [CLASS_NAMES[classes[i]] for i in range(classes.shape[0])]
        path = os.path.join(figure_path, "{0:s}.svg".format(name))
        visualization.plot_images(patches, GRID_COLUMNS, IMAGE_SIZE,
                                  "Randomly Selected Patches",
                                  subtitles=subtitles, path=path)

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

    def resize_rescale(dataset, name, factor, num_scales):
        datasets = []
        for i in range(num_scales):
            path = os.path.join(data_path, name + str(i) + "_resized")
            scale = random.triangular(SCALE_MIN, SCALE_MAX, SCALE_MODE)
            resized_rescaled = dataset.map_generator(
                make_resizer(factor*scale), path, 1)
            datasets.append(resized_rescaled)
        return datasets

    def normalize_images(examples):
        images, masks = examples
        images = preprocess.divide_median_normalize(images)
        prog.update(images.shape[0])
        return (images, masks)

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
        path = os.path.join(data_path, name + "_patched")
        patched = dataset.map_generator(extractor, path, segment_size)
        return patched

    def normalize_patches(examples):
        patches, classes = examples
        patches = preprocess.subtract_mean_normalize(patches)
        prog.update(patches.shape[0])
        return (patches, classes)

    def one_hot_encode_classes(examples):
        patches, classes = examples
        one_hot_classes = np.zeros((classes.shape[0], NUM_CLASSES))
        one_hot_classes[np.arange(one_hot_classes.shape[0]), classes] = 1
        prog.update(patches.shape[0])
        return (patches, one_hot_classes)

    # Load the datasets.
    # ==================
    data_path = os.path.join(args.outdir, "data")
    with tqdm.tqdm(desc="load datasets", total=2, unit="dataset") as prog:
        path = os.path.join(data_path, "easy_masked")
        easy_masked = dataset.Dataset(path, 1)
        path = os.path.join(repo_path, "data", "easy_masked", "data")
        easy_masked.initialize_from_aspects(path, transform_aspects)
        prog.update(1)

        path = os.path.join(data_path, "more_masked")
        more_masked = dataset.Dataset(path, 1)
        path = os.path.join(repo_path, "data", "more_masked", "data")
        more_masked.initialize_from_aspects(path, transform_aspects)
        prog.update(1)

    # Plot original images.
    # =====================
    plot_images(easy_masked, "original", "easy")
    plot_images(more_masked, "original", "more")

    # Augment the images.
    # ===================
    OFTEN, SOMETIMES, RARELY = 0.25, 0.05, 0.02

    both = iaa.Sequential([
        iaa.Fliplr(OFTEN),
        iaa.Flipud(OFTEN),
        iaa.Sometimes(SOMETIMES, iaa.PerspectiveTransform((0, 0.05)))
    ])
    inputs = iaa.Sequential([
        iaa.Invert(RARELY, per_channel=True),
        iaa.Sometimes(SOMETIMES, iaa.Add((-45, 45), per_channel=True)),
        iaa.Sometimes(RARELY,
                      iaa.AddToHueAndSaturation(value=(-15, 15),
                                                from_colorspace="RGB")),
        iaa.Sometimes(SOMETIMES, iaa.GaussianBlur(sigma=(0, 1))),
        iaa.Sometimes(RARELY,
                      iaa.Sharpen(alpha=(0, 0.25), lightness=(0.9, 1.1))),
        iaa.Sometimes(SOMETIMES,
                      iaa.AdditiveGaussianNoise(scale=(0, 0.02 * 255))),
        iaa.SaltAndPepper(SOMETIMES),
        iaa.Sometimes(SOMETIMES, iaa.ContrastNormalization((0.5, 1.5))),
        iaa.Sometimes(SOMETIMES, iaa.Grayscale(alpha=(0.0, 1.0)))
    ])
    with tqdm.tqdm(desc="augment images", total=2, unit="dataset") as prog:
        easy_masked.map_batch(make_duplicator(args.numaugs))
        easy_masked.augment(both, inputs)
        prog.update(1)

        more_masked.map_batch(make_duplicator(args.numaugs))
        more_masked.augment(both, inputs)
        prog.update(1)

    # Plot augmented images.
    # =====================
    plot_images(easy_masked, "augmented", "easy", limit=args.numaugs)
    plot_images(more_masked, "augmented", "more", limit=args.numaugs)

    # Resize the images and masks.
    # ============================
    EDGE_MODE = "reflect"
    IMAGE_ORDER = 3
    MASK_ORDER = 0
    SCALE_MIN = 0.2
    SCALE_MODE = 0.8
    SCALE_MAX = 1.5

    easy_factor = convnet1.ConvNet1.PATCH_SIZE / args.easypatchsize
    more_factor = convnet1.ConvNet1.PATCH_SIZE / args.morepatchsize

    resizes = (easy_masked.size() + more_masked.size()) * args.numscales
    with tqdm.tqdm(desc="resize images and masks", unit="example-scale",
                   total=resizes) as prog:
        easy_masked = resize_rescale(easy_masked, "easy_masked", easy_factor,
                                     args.numscales)
        more_masked = resize_rescale(more_masked, "more_masked", more_factor,
                                     args.numscales)

    # Plot resized images.
    # ====================
    for i in range(args.numscales):
        plot_images(easy_masked[i], "resized", "easy", factor=255,
                    start_num=i*easy_masked[i].size(), limit=1)
    for i in range(args.numscales):
        plot_images(more_masked[i], "resized", "more", factor=255,
                    start_num=i*more_masked[i].size(), limit=1)

    # Normalize the images.
    # =====================
    with tqdm.tqdm(desc="normalize images", total=resizes, unit="example") \
                                                                    as prog:
        for dset in easy_masked:
            dset.map_batch(normalize_images)
        for dset in more_masked:
            dset.map_batch(normalize_images)

    # Plot normalized images.
    # =============================
    for i in range(args.numscales):
        plot_images(easy_masked[i], "normalized", "easy",
                    start_num=i*easy_masked[i].size(), factor=255, limit=1)
    for i in range(args.numscales):
        plot_images(more_masked[i], "normalized", "more",
                    start_num=i*more_masked[i].size(), factor=255, limit=1)

    # Extract patches and classes.
    # ============================
    SEGMENT_PROPORTION = 0.01

    with tqdm.tqdm(desc="extract patches", total=resizes,
                   unit="example-scale") as prog:
        max_patch = int(args.maxpatches / resizes)
        for i in range(len(easy_masked)):
            easy_masked[i] = patchify(easy_masked[i],
                                      "easy_masked_{0:d}".format(i))
        for i in range(len(more_masked)):
            more_masked[i] = patchify(more_masked[i],
                                      "more_masked_{0:d}".format(i))

    # Merge the datasets.
    # ===================
    path = os.path.join(args.outdir, "masked_patched")
    size = 0
    for dset in easy_masked + more_masked:
        size += dset.size()
    segment_size = round(size * SEGMENT_PROPORTION)
    masked = dataset.Dataset(path, segment_size)
    for dset in easy_masked + more_masked:
        masked.add(dset)

    # Plot original patches.
    # ======================
    plot_patches(masked, "patched_original", factor=255)

    # Normalize the patches.
    # ======================
    with tqdm.tqdm(desc="normalize patches", total=masked.size(),
                   unit="patch") as prog:
        masked.map_batch(normalize_patches)

    # Plot normalized patches.
    # ========================
    plot_patches(masked, "patched_normalized", factor=255)

    # One-hot encode the classes.
    # ===========================
    NUM_CLASSES = 3

    with tqdm.tqdm(desc="one-hot encode classes", total=masked.size(),
                   unit="patch") as prog:
        masked.map_batch(one_hot_encode_classes)

    # Split the dataset.
    # ==================
    TEST_P = 0.1

    train_path = os.path.join(data_path, "masked_train")
    valid_path = os.path.join(data_path, "masked_validation")
    with tqdm.tqdm(desc="split dataset", total=1, unit="dataset") as prog:
        train, test = masked.split(TEST_P, train_path, valid_path)
        prog.update(1)
