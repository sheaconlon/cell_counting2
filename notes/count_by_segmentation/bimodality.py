# ========================================
# Tell Python where to find cell_counting.
# ========================================
import sys, os

root_relative_path = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, root_relative_path)

# ==========================
# Import from cell_counting.
# ==========================
from cell_counting import dataset

# ===============================
# Import from the Python library.
# ===============================
import argparse

# =================================
# Import from third-party packages.
# =================================
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # no TensorFlow logging to console

if __name__ == "__main__":
    # ===============================
    # Process command-line arguments.
    # ===============================
    parser = argparse.ArgumentParser(
        description='Investigates the bimodality of the counts_easy dataset.')
    parser.add_argument("-countseasydir", type=str, required=False,
                        default="preprocess_counts_easy_output",
                        help="A path to a directory containing the output of "
                             "preprocess_counts_easy.py.")
    parser.add_argument("-outdir", type=str, required=False,
                        default="bimodality_output",
                        help="A path to a directory in which to save output."
                             " Will be created if nonexistent.")
    parser.add_argument("-sizemin", type=int, required=False,
                        default=10, help="The minimum patch size to consider.")
    parser.add_argument("-sizemax", type=int, required=False,
                        default=100, help="The maximum patch size to consider.")
    parser.add_argument("-sizes", type=int, required=False,
                        default=10, help="The number of patch sizes to "
                                         "consider.")
    parser.add_argument("-samples", type=int, required=False,
                        default=10000, help="The number of patches to sample "
                                           "per size.")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # ===============================
    # Load the dataset.
    # ===============================
    with tqdm.tqdm(desc="load dataset", total=1, unit="datasets") as bar:
        path = os.path.join(args.countseasydir, "counts_easy_dataset")
        data = dataset.Dataset(path)
        images, counts = data.get_batch(5, 10)
        bar.update(1)

    # ===============================
    # For each example, for each size, calculate the distribution of patch
    # variances.
    # ===============================
    for example in tqdm.tqdm(range(images.shape[0]), desc="do example",
                             unit="examples"):
        image = images[example, ...]
        unique_sizes = []
        xs = []
        ys = []
        sizes = list(np.linspace(args.sizemin, args.sizemax, args.sizes))
        for i in range(len(sizes)):
            sizes[i] = round(sizes[i])
            if sizes[i] % 2 == 0:
                sizes[i] -= 1
            sizes[i] = int(sizes[i])
        for size in tqdm.tqdm(sizes, desc="do size", unit="sizes"):
            half = size // 2
            patch_ys = np.random.randint(half, image.shape[0] - half,
                                         args.samples)
            patch_xs = np.random.randint(half, image.shape[1] - half,
                                         args.samples)
            for sample in range(args.samples):
                y, x = int(patch_ys[sample]), int(patch_xs[sample])
                patch = image[y-half:y+half+1, x-half:x+half+1]
                xs.append(size)
                ys.append(np.var(patch))
        xbins = [size - 1 for size in sizes] + [sizes[-1] + 1]
        ybins = list(np.linspace(min(ys), max(ys), 30))
        plt.figure(figsize=(12, 10), dpi=300)
        plt.hist2d(xs, ys, bins=[xbins, ybins], norm=colors.LogNorm())
        plt.colorbar()
        path = os.path.join(args.outdir, "{0:d}.svg".format(example))
        plt.savefig(path, dpi='figure', format='svg')
        plt.close()
