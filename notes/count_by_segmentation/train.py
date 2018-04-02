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
    parser.add_argument("-countseasydir", type=str, required=False,
                        default="preprocess_counts_easy_output",
                        help="A path to a directory containing the output of "
                             "preprocess_counts_easy.py.")
    parser.add_argument("-outdir", type=str, required=False,
                        default="train_output",
                        help="A path to a directory in which to save output."
                             " Will be created if nonexistent.")
    parser.add_argument("-metricexamples", type=int, required=False,
                        default=10000,
                        help="The number of examples to use for each metric "
                             "evaluation.")
    parser.add_argument("-duration", type=int, required=False,
                        default=60 * 2,
                        help="The (approximate) number of minutes to train "
                             "for.")
    parser.add_argument("-metricinterval", type=int, required=False,
                        default=6,
                        help="The (approximate) number of minutes to train "
                             "between metric evaluations.")
    parser.add_argument("-countsize", type=int, required=False, default=1024,
                        help="The size to scale the largest dimension of the"
                             "counts_easy images to when counting them.")
    parser.add_argument("-mindistratio", type=float, required=False,
                        default=0.5,
                        help="The ratio between the minimum distance required"
                             " between colonies and the colony size.")
    parser.add_argument("-mindiamratio", type=float, required=False,
                        default=0.5,
                        help="The ratio between the minimum diameter required"
                             " of colonies and the colony size.")
    args = parser.parse_args()

    # =================
    # Load the dataset.
    # =================
    train_path = os.path.join(args.datadir, "masks_and_counts_train_dataset")
    valid_path = os.path.join(args.datadir,
                             "masks_and_counts_validation_dataset")
    train = dataset.Dataset(train_path)
    valid = dataset.Dataset(valid_path)

    # =====================
    # Initialize the model.
    # =====================
    SAVE_INTERVAL_FRAC = 0.5
    TQDM_PARAMS = {"desc": "initialize model", "total": 1, "unit": "model"}

    model_path = os.path.join(args.outdir, "current_model_save")
    with tqdm.tqdm(**TQDM_PARAMS) as progress_bar:
        save_interval = args.metricinterval * SAVE_INTERVAL_FRAC
        model = convnet1.ConvNet1(model_path, save_interval, train.size())
        progress_bar.update(1)

    # =======================
    # Initialize the metrics.
    # =======================
    POOL_SIZE = 10
    NUM_CLASSES = 3
    PATCH_BATCH_SIZE = 3000

    def loss_fn(actual, predicted):
        loss = losses.make_cross_entropy_loss()(actual, predicted)
        return utilities.tensor_eval(loss)

    def train_data_fn():
        return train.get_batch(args.metricexamples, POOL_SIZE)

    def valid_data_fn():
        return valid.get_batch(args.metricexamples, POOL_SIZE)

    # valid_counts_path = os.path.join(args.countseasydir,
    #                     "counts_easy_validation_dataset")
    # valid_counts_data = dataset.Dataset(valid_counts_path)
    # valid_images, valid_counts = valid_counts_data.get_all()
    # sampling_interval = int(valid_images.shape[1] / args.countsize)
    # min_dist = model.PATCH_SIZE * args.mindistratio
    # min_dist = max(1, int(min_dist / sampling_interval))
    # min_diam = model.PATCH_SIZE * args.mindiamratio
    # min_diam = min_diam / sampling_interval

    def patch_classifier(patches):
        patches = preprocess.subtract_mean_normalize(patches)
        scores = model.predict(patches)
        return scores

    # def absolute_error(model):
    #     errors = []
    #     for i in range(valid_images.shape[0]):
    #         predicted_count = postprocess.count_regions(valid_images[i, ...],
    #             model.PATCH_SIZE, patch_classifier, PATCH_BATCH_SIZE,
    #             min_dist, min_diam, sampling_interval=sampling_interval)
    #         errors.append(predicted_count - valid_counts[i])
    #     return max(errors), sum(errors) / len(errors), min(errors)
    #
    # def relative_error(model):
    #     errors = []
    #     for i in range(valid_images.shape[0]):
    #         predicted_count = postprocess.count_regions(valid_images[i, ...],
    #             model.PATCH_SIZE, patch_classifier, PATCH_BATCH_SIZE,
    #             min_dist, min_diam, sampling_interval=sampling_interval)
    #         error = (predicted_count - valid_counts[i]) / valid_counts[i]
    #         errors.append(error)
    #     return max(errors), sum(errors) / len(errors), min(errors)

    metric_path = os.path.join(args.outdir, "metrics")
    metrics = {
        "loss": metric.LossMetric(
            os.path.join(metric_path, "loss"),
            [train_data_fn, valid_data_fn], loss_fn),
        "train_confusion_matrix": metric.ConfusionMatrixMetric(
            os.path.join(metric_path, "train_confusion_matrix"),
            train_data_fn, NUM_CLASSES),
        "valid_confusion_matrix": metric.ConfusionMatrixMetric(
            os.path.join(metric_path, "valid_confusion_matrix"),
            valid_data_fn, NUM_CLASSES),
        "accuracy": metric.AccuracyMetric(
            os.path.join(metric_path, "accuracy"),
            [train_data_fn, valid_data_fn]),
        # "absolute_error": metric.DistributionMetric(
        #     os.path.join(metric_path, "absolute_error"), absolute_error),
        # "relative_error": metric.DistributionMetric(
        #     os.path.join(metric_path, "relative_error"), relative_error)
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
            path = os.path.join(iteration_path, "valid_confusion_matrix.svg")
            metrics["valid_confusion_matrix"].plot(
                "Confusion Matrix for Validation Batch", 5, 5, path=path)
            subprogress_bar.update(1)
            path = os.path.join(iteration_path, "loss.svg")
            metrics["loss"].plot("Loss",
                "number of training iterations", "loss",
                ["loss on training batch", "loss on validation batch"], 4, 10,
                path=path)
            subprogress_bar.update(1)
            path = os.path.join(iteration_path, "accuracy.svg")
            metrics["accuracy"].plot("Accuracy",
                "number of training iterations",
                "proportion of examples correctly classified",
                ["in training batch", "in validation batch"], 4, 10, path=path)
            subprogress_bar.update(1)
            # path = os.path.join(iteration_path, "absolute_error.svg")
            # metrics["absolute_error"].plot("Absolute Error",
            #     "number of training iterations", "error in count",
            #     ["most positive error", "mean error", "most negative error"],
            #     4, 10, path=path)
            # subprogress_bar.update(1)
            # path = os.path.join(iteration_path, "relative_error.svg")
            # metrics["relative_error"].plot("Relative Error",
            #     "number of training iterations", "error in count",
            #     ["most positive error", "mean error", "most negative error"],
            #     4, 10, path=path)
            # subprogress_bar.update(1)

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
