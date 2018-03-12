"""Preprocesses the ``multicondition`` dataset.

Does the following:
1. Resizes the images.
2. Normalizes the images.

Produces the following plots:
1. plate_1.svg
1. plate_2.svg
1. plate_3.svg

Saves the resulting `Dataset`s as well.

Run ``python preprocess_multicondition.py -h`` to see usage details.
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
    parser = argparse.ArgumentParser(description='Preprocess the '
                                                 'multicondition dataset.')
    parser.add_argument('-v', metavar='version', type=int, nargs=1,
                        help='a version number for the saved datasets',
                        default=1, required=False)
    args = parser.parse_args()
    version = args.v[0]

    # ======================
    # Make figure directory.
    # ======================
    FIGURE_BASE_PATH = "multicondition-{0:d}-figures".format(version)

    os.makedirs(FIGURE_BASE_PATH, exist_ok=True)

    # =================
    # Load the dataset.
    # =================
    SAVE_PATH = "multicondition-{0:d}-whole-images".format(version)
    SOURCE_PATH = "../../data/multicondition/data"
    CONDITIONS = (
        "light_uncovered_far_noperspective",
        "nolight_uncovered_close_minorperspective",
        "light_covered_close_severeperspective"
    )
    CHANNELS = ("red", "green", "blue")

    with tqdm.tqdm(desc="load images/counts") as progress_bar:
        def transform_aspects(aspects):
            channels = []
            for condition in CONDITIONS:
                for channel in CHANNELS:
                    channels.append(aspects[condition + "_" + channel])
            image = np.stack(channels, axis=2)
            count = aspects["count"]
            progress_bar.update(1)
            return (image, count)
        multicondition = dataset.Dataset(SAVE_PATH, 1)
        multicondition.initialize_from_aspects(SOURCE_PATH, transform_aspects)

    # ===================
    # Make "plate_*.svg".
    # ===================
    NUM_IMAGES = 3
    GRID_COLUMNS = 3
    IMAGE_SIZE = (4, 4)
    RGB_MAX = 255

    def separate(image):
        return [
            image[..., 0:3],
            image[..., 3:6],
            image[..., 6:9]
        ]

    images, masks = multicondition.get_batch(NUM_IMAGES)
    separated_images = []
    for i in range(NUM_IMAGES):
        separated_images.append(np.stack(separate(images[i, ...]), axis=0))
    visualization.plot_images(separated_images[0] / RGB_MAX,
                              GRID_COLUMNS,
                              IMAGE_SIZE, "Randomly Selected Plate #1",
                              path=os.path.join(FIGURE_BASE_PATH,
                                                "plate_1.svg"),
                              subtitles=list(CONDITIONS))
    visualization.plot_images(separated_images[1] / RGB_MAX,
                              GRID_COLUMNS,
                              IMAGE_SIZE, "Randomly Selected Plate #2",
                              path=os.path.join(FIGURE_BASE_PATH,
                                                "plate_2.svg"),
                              subtitles=list(CONDITIONS))
    visualization.plot_images(separated_images[2] / RGB_MAX,
                              GRID_COLUMNS,
                              IMAGE_SIZE, "Randomly Selected Plate #3",
                              path=os.path.join(FIGURE_BASE_PATH,
                                                "plate_3.svg"),
                              subtitles=list(CONDITIONS))

    # ============================
    # Resize the images and masks.
    # ============================
    ACTUAL_COLONY_DIAM = 55
    TARGET_COLONY_DIAM = 61
    RESIZE_INTERP_TYPE = "bicubic"

    resize_factor = TARGET_COLONY_DIAM / ACTUAL_COLONY_DIAM
    with tqdm.tqdm(desc="resize images", total=multicondition.size())\
            as progress_bar:
        def resize_example(example):
            image, count = example
            target_dims = tuple(round(dim*resize_factor) for dim in
                                image.shape)
            separated_images = separate(image)
            for i in range(len(separated_images)):
                separated_images[i] = misc.imresize(separated_images[i],
                                                    target_dims,
                                                    interp=RESIZE_INTERP_TYPE)
            image = np.concatenate(separated_images, axis=2)
            progress_bar.update(1)
            return [(image, count)]
        multicondition.map(resize_example)

    # =====================
    # Normalize the images.
    # =====================
    with tqdm.tqdm(desc="normalize images", total=multicondition.size())\
            as progress_bar:
        def normalize_examples(examples):
            images, counts = examples
            images = preprocess.divide_median_normalize(images)
            progress_bar.update(1)
            return (images, counts)
        multicondition.map_batch(normalize_examples)
