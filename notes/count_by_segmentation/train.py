"""Trains ``models/segmentation/convnet1`` on the ``easy`` dataset.

Produces the following plots, where * is a number of training iterations:
1. train_confusion_matrix_*.svg
2. test_confusion_matrix_*.svg
3. train_loss_*.svg
4. test_loss_*.svg

Saves the resulting `ConvNet1`.

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
from cell_counting import dataset, metric, utilities, losses, visualization
from models.segmentation.convnet1 import convnet1

# ===============================
# Import from the Python library.
# ===============================
import argparse, shutil

# =================================
# Import from third-party packages.
# =================================
import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Prevent TensorFlow's logging to console.

if __name__ == "__main__":
    # ===============================
    # Process command-line arguments.
    # ===============================
    parser = argparse.ArgumentParser(
        description='Train models/segmentation/convnet1 on the easy dataset.')
    parser.add_argument('-easyv', metavar='easy dataset version', type=int,
                        nargs=1, help='the version number of the saved easy '
                                      'dataset',
                        default=1, required=False)
    parser.add_argument('-v', metavar='version', type=int, nargs=1,
                        help='a version number for the saved model',
                        default=1, required=False)
    args = parser.parse_args()
    easy_version = args.easyv[0]
    version = args.v[0]

    # ======================
    # Make figure directory.
    # ======================
    FIGURE_BASE_PATH = "model-{0:d}-figures".format(version)

    os.makedirs(FIGURE_BASE_PATH, exist_ok=True)

    # =================
    # Load the dataset.
    # =================
    TRAIN_SAVE_PATH = "easy-{0:d}-patches-train".format(easy_version)
    TEST_SAVE_PATH = "easy-{0:d}-patches-test".format(easy_version)

    train = dataset.Dataset(TRAIN_SAVE_PATH)
    test = dataset.Dataset(TEST_SAVE_PATH)

    # =====================
    # Initialize the model.
    # =====================
    MODEL_SAVE_BASE_PATH = "model-{0:d}-saves".format(version)
    SAVE_INTERVAL = 10

    with tqdm.tqdm(desc="initialize model") as progress_bar:
        model = convnet1.ConvNet1(os.path.join(MODEL_SAVE_BASE_PATH,
                                               "current"),
                                  SAVE_INTERVAL, train.size())
        progress_bar.update(1)

    # ===================
    # Create the metrics.
    # ===================
    METRIC_SAVE_PATH = "model-{0:d}-metric-saves".format(version)
    NUM_EXAMPLES = 4_000
    NUM_CLASSES = 3

    with tqdm.tqdm(desc="get data for metrics") as progress_bar:
        metrics_train = train.get_batch(NUM_EXAMPLES)
        progress_bar.update(1)
        metrics_test = test.get_batch(NUM_EXAMPLES)
        progress_bar.update(1)
        def loss_fn(actual, pred):
            loss = losses.make_cross_entropy_loss()(actual, pred)
            return utilities.tensor_eval(loss)
        metrics = {
            "train_loss": metric.LossMetric(os.path.join(
                METRIC_SAVE_PATH, "train_loss"),
                metrics_train, loss_fn),
            "test_loss": metric.LossMetric(os.path.join(
                METRIC_SAVE_PATH, "test_loss"),
                metrics_test, loss_fn),
            "train_confusion_matrix": metric.ConfusionMatrixMetric(
                os.path.join(METRIC_SAVE_PATH, "train_confusion_matrix"),
                metrics_train, 3),
            "test_confusion_matrix": metric.ConfusionMatrixMetric(
                os.path.join(METRIC_SAVE_PATH, "test_confusion_matrix"),
                metrics_test, 3)
        }

    # ======================================================================
    # Train the model, periodically evaluating and plotting the metrics.
    # ======================================================================
    TRAIN_LENGTH = 240
    METRIC_INTERVAL = 6
    SECS_PER_MIN = 60

    def save_model():
        training_iterations = model.get_global_step() * model.get_batch_size()
        current_save = os.path.join(MODEL_SAVE_BASE_PATH, "current")
        destination_save = os.path.join(MODEL_SAVE_BASE_PATH,
                                        "at-{0:d}".format(training_iterations))
        shutil.copytree(current_save, destination_save)

    def plot_metrics():
        training_iterations = model.get_global_step() * model.get_batch_size()
        train_confusion_matrix_path = os.path.join(FIGURE_BASE_PATH,
            "train_confusion_matrix_{0:d}.svg".format(training_iterations))
        metrics["train_confusion_matrix"].plot("Training Confusion Matrix",
                                                5, 5,
                                               path=train_confusion_matrix_path)
        test_confusion_matrix_path = os.path.join(FIGURE_BASE_PATH,
            "test_confusion_matrix_{0:d}.svg".format(training_iterations))
        metrics["test_confusion_matrix"].plot("Test Confusion Matrix",
                                                5, 5,
                                                path=test_confusion_matrix_path)
        train_loss_path = os.path.join(FIGURE_BASE_PATH,
            "train_loss_{0:d}.svg".format(training_iterations))
        metrics["train_loss"].plot("Training Loss",
                                   "# of training examples seen",
                                   "loss on batch of training data", 4, 10,
                                   path=train_loss_path)
        test_loss_path = os.path.join(FIGURE_BASE_PATH,
            "test_loss_{0:d}.svg".format(training_iterations))
        metrics["test_loss"].plot("Test Loss", "# of training examples seen",
                                  "loss on batch of test data", 4, 10,
                                  path=test_loss_path)


    plot_metrics()
    for _ in tqdm.tqdm(range(TRAIN_LENGTH // METRIC_INTERVAL),
                       desc="train model/evaluate metrics"):
        model.train(train, METRIC_INTERVAL * SECS_PER_MIN)
        model.evaluate(metrics)
        plot_metrics()
        save_model()

