"""Validates the model using the validation splits of the
    ``masks_and_counts`` and ``counts_easy`` datasets.

Does the following.
1. Loads the loss metric of the final metric evaluation. Uses its data to
    determine which training step's model had the best loss on the validation
    set. Loads the model from that training step.
2. Evaluates this model's loss over the entire masks_and_counts validation set.
3. Evaluates this model's absolute and relative error distributions over the
    entire counts_easy validation set.
"""

# ========================================
# Tell Python where to find cell_counting.
# ========================================
import sys, os

root_path = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, root_path)

# ==========================
# Import from cell_counting.
# ==========================
from cell_counting import metric, dataset, losses, utilities, preprocess
from cell_counting import postprocess, visualization
from models.segmentation.convnet1 import convnet1

# ===============================
# Import from the Python library.
# ===============================
import argparse, os

# =================================
# Import from third-party packages.
# =================================
import tqdm
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # no TensorFlow logging to console

if __name__ == "__main__":
    # ===============================
    # Process command-line arguments.
    # ===============================
    parser = argparse.ArgumentParser(
        description='Validates the model using the validation splits of the'
                    'masks_and_counts and counts_easy datasets.')
    parser.add_argument("-traindir", type=str, required=False,
                        default="train_output",
                        help="A path to a directory containing the output of "
                             "train.py.")
    parser.add_argument("-maskscountsdir", type=str, required=False,
                        default="preprocess_masks_and_counts_output",
                        help="A path to a directory containing the output of "
                             "preprocess_masks_and_counts.py.")
    parser.add_argument("-countseasydir", type=str, required=False,
                        default="preprocess_counts_easy_output",
                        help="A path to a directory containing the output of "
                             "preprocess_counts_easy.py.")
    parser.add_argument("-outdir", type=str, required=False,
                        default="validate_output",
                        help="A path to a directory in which to save output."
                             " Will be created if nonexistent.")
    parser.add_argument("-countsize", type=int, required=False, default=2448,
                        help="The size to scale the largest dimension of the"
                             "counts_easy images to when counting them.")
    parser.add_argument("-mindistratio", type=float, required=False,
                        default=1/2,
                        help="The ratio between the minimum distance required"
                             " between colonies and the colony size.")
    parser.add_argument("-mindiamratio", type=float, required=False,
                        default=1/2,
                        help="The ratio between the minimum diameter required"
                             " of colonies and the colony size.")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # ==========================
    # Choose and load the model.
    # ==========================
    SAVE_INTERVAL = 5

    with tqdm.tqdm(desc="choose and load model", unit="model", total=1) \
                                                                as progress_bar:
        loss_path = os.path.join(args.traindir, "metrics", "loss")
        loss = metric.LossMetric(loss_path, None, None)
        best_iter, best_valid_loss = None, float("inf")
        iterations, valid_losses = loss._train_steps, loss._losses[:, 1]
        for iteration, valid_loss in zip(iterations, valid_losses):
            if valid_loss < best_valid_loss:
                # note: breaks tie by choosing earlier iteration
                best_iter, best_valid_loss = iteration, valid_loss
        model_path = os.path.join(args.traindir, str(best_iter), "model_save")
        model = convnet1.ConvNet1(model_path, SAVE_INTERVAL, 0)
        progress_bar.update(1)

    # ================================
    # Validate using masks_and_counts.
    # ================================
    BATCH_SIZE = 3000
    POOL_SIZE = 3

    def loss_fn(actual, predicted):
        loss = losses.make_cross_entropy_loss()(actual, predicted)
        return utilities.tensor_eval(loss)

    path = os.path.join(args.maskscountsdir,
                        "masks_and_counts_validation_dataset")
    masks_counts = dataset.Dataset(path)
    all_actual, all_predicted = [], []
    batches = masks_counts.get_batch_iterable(BATCH_SIZE, POOL_SIZE,
                                              epochs=True)

    with tqdm.tqdm(desc="validate using masks_and_counts", unit="examples",
                   total=masks_counts.size()) as progress_bar:
        while batches._epoch == 1:
            inputs, actual = next(batches)
            predicted = model.predict(inputs)
            all_actual.append(actual)
            all_predicted.append(predicted)
            progress_bar.update(BATCH_SIZE)

    actual = np.concatenate(all_actual, axis=0)
    predicted = np.concatenate(all_predicted, axis=0)
    loss = loss_fn(actual, predicted)
    f = open(os.path.join(args.outdir, "loss.csv"), "w+")
    f.write(str(float(loss)))
    f.write("\n")
    f.close()

    # ===========================
    # Validate using counts_easy.
    # ===========================
    BATCH_SIZE = 3000

    def patch_classifier(patches):
        patches = preprocess.subtract_mean_normalize(patches)
        scores = model.predict(patches)
        return scores

    path = os.path.join(args.countseasydir, "counts_easy_validation_dataset")
    counts_easy = dataset.Dataset(path)
    images, actual = counts_easy.get_all()

    sampling_interval = int(images.shape[1] / args.countsize)
    min_dist = max(1, int(args.mindistratio * model.PATCH_SIZE /
                          sampling_interval))
    min_diam = args.mindiamratio * model.PATCH_SIZE / sampling_interval

    predicted = np.empty_like(actual)
    absolute_error_sum = 0
    relative_error_sum = 0

    with tqdm.tqdm(desc="validate using counts_easy", unit="examples",
                   total=counts_easy.size()) as progress_bar:
        for i in range(images.shape[0]):
            predicted[i] = postprocess.count_regions(
                images[i, ...], model.PATCH_SIZE, patch_classifier, BATCH_SIZE,
                min_dist, min_diam, sampling_interval=sampling_interval)
            absolute_error_sum += predicted[i] - actual[i]
            relative_error_sum += (predicted[i] - actual[i]) / actual[i]
            progress_bar.update(1)

    path = os.path.join(args.outdir, "counts.svg")
    visualization.plot_scatter(actual, predicted, "Counts",
                               "Actual count (CFU)", "Predicted count (CFU)",
                               4, 10, path=path)
    absolute_error_avg = absolute_error_sum / images.shape[0]
    relative_error_avg = relative_error_sum / images.shape[0]
    f = open(os.path.join(args.outdir, "absolute_error_average.csv"), "w+")
    f.write(str(absolute_error_avg))
    f.write("\n")
    f.close()
    f = open(os.path.join(args.outdir, "relative_error_average.csv"), "w+")
    f.write(str(relative_error_avg))
    f.write("\n")
    f.close()
