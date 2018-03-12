"""Trains ``models/segmentation/convnet1`` on the ``easy`` dataset.

Produces the following plots:

Saves the resulting `ConvNet1` to disk.

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
import argparse

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
    parser.add_argument('-dv', metavar='dataset version', type=int, nargs=1,
                        help='the version number of the dataset to use',
                        default=1, required=False)
    parser.add_argument('-v', metavar='version', type=int, nargs=1,
                        help='a version number to save the model with',
                        default=1, required=False)
    args = parser.parse_args()
    data_version = args.dv[0]
    version = args.v[0]

    # ======================
    # Make figure directory.
    # ======================
    FIGURE_BASE_PATH = "figures-{0:d}".format(version)

    os.makedirs(FIGURE_BASE_PATH, exist_ok=True)

    # =================
    # Load the dataset.
    # =================
    TRAIN_SAVE_PATH = "easy-{0:d}-train".format(data_version)
    TEST_SAVE_PATH = "easy-{0:d}-test".format(data_version)

    train = dataset.Dataset(TRAIN_SAVE_PATH)
    test = dataset.Dataset(TEST_SAVE_PATH)

    # =====================
    # Initialize the model.
    # =====================
    SAVE_INTERVAL = 10

    with tqdm.tqdm(desc="initialize model") as progress_bar:
        model = convnet1.ConvNet1("segmentation-convnet1-{0:d}".format(
            version), SAVE_INTERVAL, train.size())
        progress_bar.update(1)

    # ===================
    # Create the metrics.
    # ===================
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
            "train_loss": metric.LossMetric(metrics_train, loss_fn),
            "test_loss": metric.LossMetric(metrics_test, loss_fn)
        }

    # =====================================================
    # Train the model, periodically evaluating the metrics.
    # =====================================================
    TRAIN_LENGTH = 240
    METRIC_INTERVAL = 6
    SECS_PER_MIN = 60

    for _ in tqdm.tqdm(range(TRAIN_LENGTH // METRIC_INTERVAL),
                       desc="train model/evaluate metrics"):
        model.train(train, METRIC_INTERVAL * SECS_PER_MIN)
        model.evaluate(metrics)

    # =================
    # Plot the metrics.
    # =================
    metrics["train_loss"].plot("Training Loss", "# of training examples seen",
                               "loss on batch of training data", 4,  10,
                               path=os.path.join(FIGURE_BASE_PATH,
                                                 "train_loss.svg"))
    metrics["test_loss"].plot("Test Loss", "# of training examples seen",
                              "loss on batch of test data", 4, 10,
                              path=os.path.join(FIGURE_BASE_PATH,
                                                "test_loss.svg"))
