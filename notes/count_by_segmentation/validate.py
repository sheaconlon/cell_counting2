"""Validates the model using the validation splits of the ``easy_masked``
    dataset.

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
from cell_counting import metric, dataset, losses, utilities
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
        description='Validates the model using the validation split of the'
                    'easy_masked dataset.')
    parser.add_argument("-traindir", type=str, required=False,
                        default="train",
                        help="A path to a directory containing the output of "
                             "train.py.")
    parser.add_argument("-easymaskeddir", type=str, required=False,
                        default="preprocess_easy_masked",
                        help="A path to a directory containing the output of "
                             "preprocess_easy_masked.py.")
    parser.add_argument("-outdir", type=str, required=False,
                        default="validate",
                        help="A path to a directory in which to save output."
                             " Will be created if nonexistent.")
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
        f = open(os.path.join(args.outdir, "best_iteration.csv"), "w+")
        f.write(str(best_iter))
        f.write("\n")
        f.close()
        model_path = os.path.join(args.traindir, str(best_iter), "model_save")
        model = convnet1.ConvNet1(model_path, SAVE_INTERVAL, 0)
        progress_bar.update(1)

    # ================================
    # Validate using masks_and_counts.
    # ================================
    BATCH_SIZE = 3000
    POOL_SIZE = 10

    def loss_fn(actual, predicted):
        loss = losses.make_cross_entropy_loss()(actual, predicted)
        return utilities.tensor_eval(loss)

    path = os.path.join(args.maskscountsdir, "easy_masked_validation")
    masks_counts = dataset.Dataset(path)
    all_actual, all_predicted = [], []
    batch_size = min(BATCH_SIZE, masks_counts.size())
    batches = masks_counts.get_batch_iterable(batch_size, POOL_SIZE,
                                              epochs=True)

    with tqdm.tqdm(desc="validate using easy_masked", unit="examples",
                   total=masks_counts.size()) as progress_bar:
        inputs, actual = next(batches)
        while batches._epoch == 1:
            predicted = model.predict(inputs)
            all_actual.append(actual)
            all_predicted.append(predicted)
            progress_bar.update(inputs.shape[0])
            inputs, actual = next(batches)

    actual = np.concatenate(all_actual, axis=0)
    predicted = np.concatenate(all_predicted, axis=0)
    loss = loss_fn(actual, predicted)
    f = open(os.path.join(args.outdir, "loss.csv"), "w+")
    f.write(str(float(loss)))
    f.write("\n")
    f.close()
