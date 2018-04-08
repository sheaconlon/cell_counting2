"""Tests the model using the ``masks_and_counts`` training/validation splits,
    the ``counts_easy`` test split, ``counts_multicondition``, and ``more``.

Produces the following plots.
1. pixels/confusion_matrix.svg
2. pixels/confusion_examples/predicted-*/actual-*/+.svg for * any of "inside",
    "edge", or "outside" and + an number
3. pixels/loss.svg
4. pixels/accuracy_vs_patch_size.svg
5. counts/predicted_vs_actual.svg
6. counts/overlay_images/*.svg for * an example number
7. counts/inside_images/*.svg for * an example number
8. counts/distance_images/*.svg for * an example number
9. counts/marker_images/*.svg for * an example number
10. counts/label_images/*.svg for * an example number
11. counts/actual_counts/*.txt for * an example number
12. counts/predicted_counts/*.txt for * an example number

Run ``python test.py -h`` to see usage details.
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
from cell_counting import dataset, postprocess, visualization, preprocess
from cell_counting import metric
from models.segmentation.convnet1 import convnet1

# ===============================
# Import from the Python library.
# ===============================
import argparse, random, subprocess, math, shutil

# =================================
# Import from third-party packages.
# =================================
import tqdm
import numpy as np
import imageio
from skimage import transform
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # no TensorFlow logging to console

if __name__ == "__main__":
    # ===============================
    # Process command-line arguments.
    # ===============================
    parser = argparse.ArgumentParser(
        description='Validates the model using the validation splits of the'
                    'masks_and_counts and counts_easy datasets.')
    parser.add_argument("-maskscountsdir", type=str, required=False,
                        default="preprocess_masks_and_counts_output",
                        help="A path to a directory containing the output of "
                             "preprocess_masks_and_counts.py.")
    parser.add_argument("-countseasydir", type=str, required=False,
                        default="preprocess_counts_easy_output",
                        help="A path to a directory containing the output of "
                             "preprocess_counts_easy.py.")
    parser.add_argument("-countsmulticonditiondir", type=str, required=False,
                        default="preprocess_counts_multicondition_output",
                        help="A path to a directory containing the output of "
                             "preprocess_counts_multicondition.py.")
    parser.add_argument("-moredir", type=str, required=False,
                        default="preprocess_more_output",
                        help="A path to a directory containing the output of "
                             "preprocess_more.py.")
    parser.add_argument("-traindir", type=str, required=False,
                        default="train_output",
                        help="A path to a directory containing the output of "
                             "train.py.")
    parser.add_argument("-validatedir", type=str, required=False,
                        default="validate_output",
                        help="A path to a directory containing the output of "
                             "validate.py.")
    parser.add_argument("-outdir", type=str, required=False,
                        default="test_output",
                        help="A path to a directory in which to save output."
                             " Will be created if nonexistent.")
    parser.add_argument("-countsize", type=int, required=False, default=1024,
                        help="The size to scale the largest dimension of"
                             " images to when counting them.")
    parser.add_argument("-mindistratio", type=float, required=False,
                        default=1/2,
                        help="The ratio between the minimum distance required"
                             " between colonies and the colony size.")
    parser.add_argument("-mindiamratio", type=float, required=False,
                        default=1/2,
                        help="The ratio between the minimum diameter required"
                             " of colonies and the colony size.")
    args = parser.parse_args()
    os.makedirs(os.path.join(args.outdir, "pixels"), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "counts"), exist_ok=True)

    # ==================
    # Load the datasets.
    # ==================
    TQDM_PARAMS = {"desc": "load datasets", "total": 5, "unit": "dataset"}

    with tqdm.tqdm(**TQDM_PARAMS) as progress_bar:
        masks_counts_train_path = os.path.join(args.maskscountsdir,
                                  "masks_and_counts_train_dataset")
        masks_counts_train = dataset.Dataset(masks_counts_train_path)
        progress_bar.update(1)

        masks_counts_valid_path = os.path.join(args.maskscountsdir,
                                 "masks_and_counts_validation_dataset")
        masks_counts_valid = dataset.Dataset(masks_counts_valid_path)
        progress_bar.update(1)

        counts_easy_path = os.path.join(args.countseasydir,
                                        "counts_easy_test_dataset")
        counts_easy = dataset.Dataset(counts_easy_path)
        progress_bar.update(1)

        counts_multicondition_path = os.path.join(args.countsmulticonditiondir,
                                                  "counts_multicondition_dataset")
        counts_multicondition = dataset.Dataset(counts_multicondition_path)
        progress_bar.update(1)

        more_path = os.path.join(args.moredir, "more_dataset")
        more = dataset.Dataset(more_path)
        progress_bar.update(1)

    # =================
    # Set up the model.
    # =================
    TQDM_PARAMS = {"desc": "set up model", "total": 1, "unit": "model"}

    with tqdm.tqdm(**TQDM_PARAMS) as progress_bar:
        path = os.path.join(args.validatedir, "best_iteration.csv")
        with open(path, "r") as f:
            best_iter = int(f.readline()[:-1])
        path = os.path.join(args.traindir, str(best_iter), "model_save")
        model = convnet1.ConvNet1(path, 0, 0)
        progress_bar.update(1)

    # ========================================
    # Create the following.
    # - "counts/predicted_vs_actual.svg"
    # - "counts/overlay_images/*.svg"
    # - "counts/inside_images/*.svg"
    # - "counts/distance_images/*.svg"
    # - "counts/marker_images/*.svg"
    # - "counts/label_images/*.svg"
    # - "counts/actual_counts/*.txt"
    # - "counts/predicted_counts/*.txt"
    # ========================================
    BATCH_SIZE = 3000
    TQDM_PARAMS = {"desc": "count", "unit": "datasets", "total": 5}
    CONDITIONS = (
        "light_uncovered_far_noperspective",
        "nolight_uncovered_close_minorperspective",
        "light_covered_close_severeperspective"
    )

    results = {}

    def make_predictions(name, images, counts):
        def classifier(patches):
            patches = preprocess.subtract_mean_normalize(patches)
            scores = model.predict(patches)
            subprogress_bar.update(BATCH_SIZE)
            return scores

        results[name] = []

        max_dim = max(images.shape[1], images.shape[2])
        sampling = max(1, int(max_dim / args.countsize))
        min_dist = int(model.PATCH_SIZE / sampling * args.mindistratio)
        min_dist = max(1, min_dist)
        min_diam = model.PATCH_SIZE / sampling * args.mindiamratio

        approx_patches = (images.shape[1] // sampling) * \
                         (images.shape[2] // sampling)
        tqdm_params = {"desc": "count {0:s}".format(name), "unit": "images",
                       "total": images.shape[0]}
        for i in tqdm.tqdm(range(images.shape[0]), **tqdm_params):
            tqdm_params = {"desc": "count image {0:d}".format(i),
                           "unit": "patches", "total": approx_patches}
            with tqdm.tqdm(**tqdm_params) as subprogress_bar:
                image = images[i, ...]
                predicted, image_dict = postprocess.count_regions(image,
                                            model.PATCH_SIZE, classifier,
                                            BATCH_SIZE, min_dist, min_diam,
                                            sampling_interval=sampling,
                                            debug=True)
                results[name].append((counts[i], predicted, image_dict))

    with tqdm.tqdm(**TQDM_PARAMS) as progress_bar:
        images, counts = counts_easy.get_all()
        make_predictions("counts_easy test split", images, counts)
        progress_bar.update(1)

        image_sets, counts = counts_multicondition.get_all()
        for i, condition in enumerate(CONDITIONS):
            name = "counts_multicondition {0:s} condition".format(condition)
            images = image_sets[:, i, ...]
            make_predictions(name, images, counts)
            progress_bar.update(1)

        images, counts = more.get_all()
        make_predictions("more", images, counts)
        progress_bar.update(1)

    def write_image(path, image):
        RGB_MAX = 255

        os.makedirs(os.path.split(path)[0], exist_ok=True)
        image = image.astype(np.float64)
        image = image - np.amin(image)
        image = image * RGB_MAX / np.amax(image)
        image = image.astype(np.uint8)
        imageio.imwrite(path, image)

    xs = []
    ys = []
    colors = []
    i = 0
    with tqdm.tqdm(desc="writing images", unit="datasets", total=5) as bar:
        for result_list in results.values():
            color = random.random()
            with tqdm.tqdm(desc="writing images for dataset", unit="examples",
                           total=len(result_list)) as bar2:
                for actual, predicted, image_dict in result_list:
                    xs.append(actual)
                    ys.append(predicted)
                    colors.append(color)
                    path = os.path.join(args.outdir, "counts", "inside_images",
                                        "{0:d}.png".format(i))
                    write_image(path, image_dict["inside"])
                    path = os.path.join(args.outdir, "counts", "distance_images",
                                        "{0:d}.png".format(i))
                    write_image(path, image_dict["distance"])
                    path = os.path.join(args.outdir, "counts", "marker_images",
                                        "{0:d}.png".format(i))
                    write_image(path, image_dict["marker"])
                    path = os.path.join(args.outdir, "counts", "label_images",
                                        "{0:d}.png".format(i))
                    write_image(path, image_dict["label"])
                    overlay = np.copy(image_dict["original"] * 255)
                    inside = np.zeros(overlay.shape[:2])
                    half = model.PATCH_SIZE // 2
                    new_dims = (inside.shape[0] - 2*half,
                                inside.shape[1] - 2*half)
                    inside[half:-half, half:-half] = transform.resize(
                        image_dict["inside"].astype(np.float64) * 255, new_dims,
                        order=3, mode="reflect", clip=False)
                    overlay[inside > 0, :] = np.array([255, 0, 0])[np.newaxis,
                                                                       np.newaxis, :]
                    path = os.path.join(args.outdir, "counts", "overlay_images",
                                        "{0:d}.png".format(i))
                    write_image(path, overlay)
                    i += 1
                    bar2.update(1)
            bar.update(1)

    path = os.path.join(args.outdir, "counts", "predicted_vs_actual.svg")
    visualization.plot_scatter(xs, ys, "CFU Counts",
        "actual count (CFU)", "predicted count (CFU)", 4, 10,
        colors=colors, path=path)

    # ===========================================
    # Create "pixels/accuracy_vs_patch_size.svg".
    # ===========================================
    SIZE_DEVIATION = 2
    ACTUAL_SIZE = 43
    SIZES = 10
    PATCHES = 10000
    BATCH = 3000
    POOL = 10

    def loss_fn(actual, predicted):
        actual = np.argmax(actual, axis=1)
        predicted = np.argmax(predicted, axis=1)
        return np.mean(np.equal(actual, predicted))

    min_size = max(1, math.floor(ACTUAL_SIZE/SIZE_DEVIATION))
    max_size = math.ceil(ACTUAL_SIZE*SIZE_DEVIATION)
    sizes = []
    accuracies = []
    with tqdm.tqdm(desc="create pixels/accuracy_vs_patch_size.svg",
                   unit="patch sizes", total=SIZES) as bar:
        for size in np.geomspace(min_size, max_size, SIZES):
            size = round(size)
            subprocess.call(["python3", "../preprocess_masks_and_counts.py",
                             "-maxpatch", str(PATCHES), "-patchsize",
                             str(size), "-outdir", "test_tmp"])
            data = dataset.Dataset(
                "test_tmp/masks_and_counts_train_dataset")
            all_actual, all_predicted = [], []
            batch = min(BATCH, data.size())
            batches = data.get_batch_iterable(batch, POOL, epochs=True)

            with tqdm.tqdm(desc="assess accuracy for size {0:f}".format(size),
                           unit="examples", total=data.size()) as bar2:
                inputs, actual = next(batches)
                while batches._epoch == 1:
                    predicted = model.predict(inputs)
                    all_actual.append(actual)
                    all_predicted.append(predicted)
                    bar2.update(inputs.shape[0])
                    inputs, actual = next(batches)
            actual = np.concatenate(all_actual, axis=0)
            predicted = np.concatenate(all_predicted, axis=0)
            actual = np.argmax(actual, axis=1)
            predicted = np.argmax(predicted, axis=1)
            accuracy = np.mean(np.equal(actual, predicted))
            sizes.append(size)
            accuracies.append(accuracy)
            shutil.rmtree("test_tmp")
            bar.update(1)

    path = os.path.join(args.outdir, "pixels", "accuracy_vs_patch_size.svg")
    visualization.plot_scatter(sizes, accuracies, "Effect of Patch Size at "
                               "Testing Time", "patch size (px)",
                               "pixelwise classification accuracy", 4, 10,
                               path=path)

