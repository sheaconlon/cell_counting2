"""Trains ``models/segmentation/convnet1`` on the ``counts_and_masks`` dataset.

Produces the following plots, where * is a number of training iterations:
1. */train_confusion_matrix.svg
2. */test_confusion_matrix.svg
3. */loss.svg
4. */accuracy.svg

Saves `ConvNet1`s in `*/model_save`, where * is a number of training iterations.
    The current `ConvNet1` is saved in `current_model_save`.

Run ``python train.py -h`` to see usage details.
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
from cell_counting import dataset, metric, utilities, losses, preprocess, \
    postprocess
from models.segmentation.convnet1 import convnet1

# ===============================
# Import from the Python library.
# ===============================
import argparse, shutil

# =================================
# Import from third-party packages.
# =================================
import tqdm

# Prevent TensorFlow from logging to the console.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == "__main__":
    # ===============================
    # Process command-line arguments.
    # ===============================
    parser = argparse.ArgumentParser(
        description='Train models/segmentation/convnet1 on the '
                    'counts_and_masks dataset.')
    parser.add_argument("-datadir", type=str, required=False,
                        default="preprocess_masks_and_counts_output",
                        help="A path to a directory containing the output of "
                             "preprocess_masks_and_counts.py.")
    parser.add_argument("-multiconditionpiecesdir", type=str, required=False,
                        default="preprocess_multicondition_pieces_output",
                        help="A path to a directory containing the output of "
                             "preprocess_multicondition_pieces.py.")
    parser.add_argument("-outdir", type=str, required=False,
                        default="train_output",
                        help="A path to a directory in which to save output."
                             " Will be created if nonexistent.")
    parser.add_argument("-metricexamples", type=int, required=False,
                        default=3000,
                        help="The number of examples to use for each metric "
                             "evaluation.")
    parser.add_argument("-duration", type=int, required=False,
                        default=60 * 4,
                        help="The (approximate) number of minutes to train "
                             "for.")
    parser.add_argument("-metricinterval", type=int, required=False,
                        default=3,
                        help="The (approximate) number of minutes to train "
                             "between metric evaluations.")
    args = parser.parse_args()

    # =================
    # Load the dataset.
    # =================
    train_path = os.path.join(args.datadir, "masks_and_counts_train_dataset")
    test_path = os.path.join(args.datadir, "masks_and_counts_test_dataset")
    train = dataset.Dataset(train_path)
    test = dataset.Dataset(test_path)

    # =====================
    # Initialize the model.
    # =====================
    SAVE_INTERVAL = 5
    TQDM_PARAMS = {"desc": "initialize model", "total": 1, "unit": "model"}

    model_path = os.path.join(args.outdir, "current_model_save")
    with tqdm.tqdm(**TQDM_PARAMS) as progress_bar:
        model = convnet1.ConvNet1(model_path, SAVE_INTERVAL, train.size())
        progress_bar.update(1)

    # =======================
    # Initialize the metrics.
    # =======================
    POOL_SIZE = 5
    NUM_CLASSES = 3
    PATCH_BATCH_SIZE = 3000
    MIN_DIST_FRAC = 1 / 2
    MIN_DIAM_FRAC = 1 / 2
    SAMPLING_TARGET = 400 # the original size of the pieces

    def loss_fn(actual, predicted):
        loss = losses.make_cross_entropy_loss()(actual, predicted)
        return utilities.tensor_eval(loss)

    def train_data_fn():
        return train.get_batch(args.metricexamples, POOL_SIZE)

    def test_data_fn():
        return test.get_batch(args.metricexamples, POOL_SIZE)

    path = os.path.join(args.multiconditionpiecesdir,
                        "multicondition_pieces_dataset")
    pieces_dataset = dataset.Dataset(path)
    pieces_images, pieces_counts = pieces_dataset.get_all()
    sampling_interval = int(pieces_images.shape[1] / SAMPLING_TARGET)
    min_dist = model.PATCH_SIZE * MIN_DIST_FRAC
    min_dist = max(1, int(min_dist / sampling_interval))
    min_diam = model.PATCH_SIZE * MIN_DIAM_FRAC
    min_diam = min_diam / sampling_interval

    def patch_classifier(patches):
        patches = preprocess.subtract_mean_normalize(patches)
        scores = model.predict(patches)
        return scores

    def absolute_error(model):
        errors = []
        for i in range(pieces_images.shape[0]):
            predicted_count = postprocess.count_regions(pieces_images[i, ...],
                model.PATCH_SIZE, patch_classifier, PATCH_BATCH_SIZE,
                min_dist, min_diam, sampling_interval=sampling_interval)
            errors.append(predicted_count - pieces_counts[i])
        return max(errors), sum(errors) / len(errors), min(errors)

    def relative_error(model):
        errors = []
        for i in range(pieces_images.shape[0]):
            predicted_count = postprocess.count_regions(pieces_images[i, ...],
                model.PATCH_SIZE, patch_classifier, PATCH_BATCH_SIZE,
                min_dist, min_diam, sampling_interval=sampling_interval)
            error = (predicted_count - pieces_counts[i]) / pieces_counts[i]
            errors.append(error)
        return max(errors), sum(errors) / len(errors), min(errors)

    metric_path = os.path.join(args.outdir, "metrics")
    metrics = {
        "loss": metric.LossMetric(
            os.path.join(metric_path, "loss"),
            [train_data_fn, test_data_fn], loss_fn),
        "train_confusion_matrix": metric.ConfusionMatrixMetric(
            os.path.join(metric_path, "train_confusion_matrix"),
            train_data_fn, NUM_CLASSES),
        "test_confusion_matrix": metric.ConfusionMatrixMetric(
            os.path.join(metric_path, "test_confusion_matrix"),
            test_data_fn, NUM_CLASSES),
        "accuracy": metric.AccuracyMetric(
            os.path.join(metric_path, "accuracy"),
            [train_data_fn, test_data_fn]),
        "absolute_error": metric.DistributionMetric(
            os.path.join(metric_path, "absolute_error"), absolute_error),
        "relative_error": metric.DistributionMetric(
            os.path.join(metric_path, "relative_error"), relative_error)
    }

    # ==================================================================
    # Train the model, periodically evaluating and plotting the metrics.
    # ==================================================================
    SECS_PER_MIN = 60

    def save_model():
        TQDM_PARAMS = {"desc": "save model", "unit": "model", "total": 1,
                       "leave": False}
        with tqdm.tqdm(**TQDM_PARAMS) as subprogress_bar:
            training_iterations = model.get_global_step()
            iteration_path = os.path.join(args.outdir, str(training_iterations))
            save_path = os.path.join(iteration_path, "model_save")
            shutil.copytree(model_path, save_path)
            subprogress_bar.update(1)

    def plot_metrics():
        tqdm_params = {"desc": "plot_metrics", "unit": "metric", "leave": False,
                       "total": len(metrics)}
        with tqdm.tqdm(**tqdm_params) as subprogress_bar:
            training_iterations = model.get_global_step()
            iteration_path = os.path.join(args.outdir, str(training_iterations))
            os.makedirs(iteration_path, exist_ok=True)
            path = os.path.join(iteration_path, "train_confusion_matrix.svg")
            metrics["train_confusion_matrix"].plot(
                "Confusion Matrix for Training Batch", 5, 5, path=path)
            subprogress_bar.update(1)
            path = os.path.join(iteration_path, "test_confusion_matrix.svg")
            metrics["test_confusion_matrix"].plot(
                "Confusion Matrix for Test Batch", 5, 5, path=path)
            subprogress_bar.update(1)
            path = os.path.join(iteration_path, "loss.svg")
            metrics["loss"].plot("Loss",
                "number of training iterations", "loss",
                ["loss on training batch", "loss on test batch"], 4, 10,
                path=path)
            subprogress_bar.update(1)
            path = os.path.join(iteration_path, "accuracy.svg")
            metrics["accuracy"].plot("Accuracy",
                "number of training iterations",
                "proportion of examples correctly classified",
                ["in training batch", "in test batch"], 4, 10, path=path)
            subprogress_bar.update(1)
            path = os.path.join(iteration_path, "absolute_error.svg")
            metrics["absolute_error"].plot("Absolute Error",
                "number of training iterations", "error in count",
                ["most positive error", "mean error", "most negative error"],
                4, 10, path=path)
            subprogress_bar.update(1)
            path = os.path.join(iteration_path, "relative_error.svg")
            metrics["relative_error"].plot("Relative Error",
                "number of training iterations", "error in count",
                ["most positive error", "mean error", "most negative error"],
                4, 10, path=path)
            subprogress_bar.update(1)

    def callback():
        TQDM_PARAMS = {"desc": "evaluate model", "unit": "model", "total": 1,
                       "leave": False}
        with tqdm.tqdm(**TQDM_PARAMS) as subprogress_bar:
            model.evaluate(metrics)
            subprogress_bar.update(1)
        save_model()
        plot_metrics()
        progress_bar.update(1)


    tqdm_params = {"desc": "train model", "unit": "round",
                   "total": args.duration // args.metricinterval + 1}
    with tqdm.tqdm(**tqdm_params) as progress_bar:
        model.train_epochs(train, args.duration, callback, args.metricinterval)
