import os

import imageio
import numpy as np


def load():
    """Yields examples from this dataset."""
    data_path = os.path.join(os.path.dirname(__file__), "data")
    for example_name in os.listdir(data_path):
        if example_name[0] == '.':
            continue
        example_path = os.path.join(data_path, example_name)
        image = imageio.imread(os.path.join(example_path, "image.png"))
        count = np.loadtxt(os.path.join(example_path, "count.csv"),
                           delimiter=',')
        yield (image, count)
