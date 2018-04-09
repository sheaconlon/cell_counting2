"""Trains ``models/segmentation/convnet1`` on the ``easy_masked`` dataset.

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
        description='Train models/segmentation/convnet1 on the easy_masked'
                    ' dataset.')
    parser.add_argument("-easymaskeddir", type=str, required=False,
                        default="preprocess_easy_masked",
                        help="A path to a directory containing the output of "
                             "preprocess_easy_masked.py.")
    parser.add_argument("-outdir", type=str, required=False,
                        default="train",
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
    args = parser.parse_args()

    # =================
    # Load the dataset.
    # =================
    train_path = os.path.join(args.easymaskeddir, "easy_masked_train")
    valid_path = os.path.join(args.easymaskeddir, "easy_masked_validation")
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

    def patch_classifier(patches):
        patches = preprocess.subtract_mean_normalize(patches)
        scores = model.predict(patches)
        return scores

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
