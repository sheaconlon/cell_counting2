import os

import imageio
import numpy as np

CONDITIONS = (
    "light_uncovered_far_noperspective",
    "nolight_uncovered_close_minorperspective",
    "light_covered_close_severeperspective"
)

def load():
    """Yields examples from this dataset."""
    data_path = os.path.join(os.path.dirname(__file__), "data")
    for example_name in os.listdir(data_path):
        if example_name[0] == '.':  # skip hidden files -- they are junk
            continue
        example_path = os.path.join(data_path, example_name)
        images = []
        for condition in CONDITIONS:
            image_path = os.path.join(example_path,
                                      "{0:s}.png".format(condition))
            image = imageio.imread(image_path)
            images.append(image)
        image_set = np.stack(images, axis=0)
        count_path = os.path.join(example_path, "count.csv")
        count = np.loadtxt(count_path, delimiter=',')
        yield image_set, count