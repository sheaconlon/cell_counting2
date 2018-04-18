"""Tests the model on the ``pinned`` dataset.

Produces the following output, where <EXAMPLENUM> is replaced by each example
number.
1. <OUT>/counts.csv
2. <OUT>/predicted_vs_actual.svg
3. <OUT>/masks/<EXAMPLENUM>/1_inside.svg
4. <OUT>/masks/<EXAMPLENUM>/2_distance.svg
5. <OUT>/masks/<EXAMPLENUM>/3_peak.svg
6. <OUT>/masks/<EXAMPLENUM>/4_marker.svg
7. <OUT>/masks/<EXAMPLENUM>/5_label.svg

Run ``python test_pinned.py -h`` to see usage details.
"""

# ========================================
# Tell Python where to find cell_counting.
# ========================================
import os
import sys

path = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, path)

# ==========================
# Import from cell_counting.
# ==========================
from cell_counting import dataset, preprocess, postprocess, visualization
from cell_counting.models.segmentation import convnet1

# ===============================
# Import from the Python library.
# ===============================
import argparse

# =================================
# Import from third-party packages.
# =================================
import tqdm
import numpy as np
import imageio

# prevent TensorFlow logging to console
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == "__main__":
    # ===============================
    # Process command-line arguments.
    # ===============================
    parser = argparse.ArgumentParser(
        description="Tests the model on the 'pinned' dataset.")
    parser.add_argument("-out", type=str, required=False,
                        default="test_pinned",
                        help="A path to a directory in which to save output."
                             " Will be created if nonexistent.")
    parser.add_argument("-mindist", type=float, required=False, default=1/2,
                        help="The minimum distance between colonies, expressed"
                             " as a factor of the colony size.")
    parser.add_argument("-mindiam", type=float, required=False, default=1/2,
                        help="The minimum diameter of colonies, expressed as a"
                             " factor of the colony size.")
    parser.add_argument("-pinned", type=str, required=False,
                        default="preprocess_pinned",
                        help="A path to the output of preprocess_pinned.py.")
    parser.add_argument("-train", type=str, required=False, default="train",
                        help="A path to the output of train.py.")
    parser.add_argument("-valid", type=str, required=False,
                        default="validate",
                        help="A path to the output of validate.py.")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # =================
    # Load the dataset.
    # =================
    with tqdm.tqdm(desc="Loading dataset", total=1, unit="datasets") as prog:
        path = os.path.join(args.pinned, "pinned")
        pinned = dataset.Dataset(path)
        prog.update(1)

    # =================
    # Set up the model.
    # =================
    with tqdm.tqdm(desc="Setting up model", total=1, unit="models") as prog:
        path = os.path.join(args.valid, "best_iteration.csv")
        with open(path, "r") as f:
            best_iter = int(f.readline()[:-1])
        path = os.path.join(args.train, str(best_iter), "model_save")
        model = convnet1.ConvNet1(path, 0, 0)
        prog.update(1)

    # ===============
    # Count 'pinned'.
    # ===============
    BATCH_SIZE = 4000
    MASKS = ("inside", "distance", "peak", "marker", "label")

    def classifier(patches):
        patches = preprocess.subtract_mean_normalize(patches)
        scores = model.predict(patches)
        prog2.update(patches.shape[0])
        return scores

    def save_image(path, image):
        RGB_MAX = 255

        os.makedirs(os.path.split(path)[0], exist_ok=True)
        image = image.astype(np.float64)
        image = image - np.amin(image)
        image = image * RGB_MAX / np.amax(image)
        image = image.astype(np.uint8)
        imageio.imwrite(path, image)

    def save_counts():
        counts_arr = np.stack(count_data)
        path = os.path.join(args.out, "counts.csv")
        np.savetxt(path, counts_arr, fmt="%d", delimiter=",")

    def plot_counts():
        xs = [row[1] for row in count_data]
        ys = [row[2] for row in count_data]
        path = os.path.join(args.out, "predicted_vs_actual.svg")
        visualization.plot_scatter(xs, ys,
                                   "CFU Counts for 'pinned' Dataset"
                                   " Well Images", "actual count (CFU)",
                                   "predicted count (CFU)", 4, 10, path=path)

    images, counts = pinned.get_all()
    count_data = []
    for i in tqdm.trange(images.shape[0], desc="Counting 'pinned'",
                         unit="well images"):
        image, count = images[i, ...], counts[i]
        with tqdm.tqdm(desc="Counting well image #{0:d}".format(i),
                       total=image.shape[0]*image.shape[1],
                       unit="patches") as prog2:
            mindist = int(args.mindist*model.PATCH_SIZE)
            mindiam = args.mindiam*model.PATCH_SIZE
            predicted, masks = postprocess.count_regions(
                image, model.PATCH_SIZE, classifier, BATCH_SIZE,
                mindist, mindiam, debug=True)
            count_data.append(np.array([i, count, predicted]))
        with tqdm.tqdm(desc="Saving outputs for well image #{0:d}".format(i),
                       total=2+len(MASKS), unit="outputs") as prog2:
            save_counts()
            prog2.update(1)
            plot_counts()
            prog2.update(1)
            mask_base = "{0:s}/masks/{1:d}".format(args.out, i)
            for num, mask in enumerate(MASKS):
                name = "{0:d}_{1:s}.png".format(num + 1, mask)
                path = os.path.join(mask_base, name)
                save_image(path, masks[mask])
                prog2.update(1)
