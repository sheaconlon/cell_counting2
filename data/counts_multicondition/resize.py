import os
import shutil

from skimage import transform
import imageio
import numpy as np

CONDITIONS = (
    "light_uncovered_far_noperspective",
    "nolight_uncovered_close_minorperspective",
    "light_covered_close_severeperspective"
)
SIZE = (2448, 2448)
ORDER = 3
EDGE_MODE = "reflect"

if __name__ == "__main__":
    raw_path = os.path.join(os.path.dirname(__file__), "raw")
    data_path = os.path.join(os.path.dirname(__file__), "data")
    shutil.rmtree(data_path)
    for example_name in os.listdir(raw_path):
        if example_name[0] == '.':  # skip hidden files -- they are junk
            continue
        raw_example_path = os.path.join(raw_path, example_name)
        example_path = os.path.join(data_path, example_name)
        os.makedirs(example_path, exist_ok=True)
        count_path = os.path.join(raw_example_path, "count.csv")
        shutil.copy(count_path, example_path)
        for condition in CONDITIONS:
            image_path = os.path.join(raw_example_path,
                                      "{0:s}.jpg".format(condition))
            image = imageio.imread(image_path)
            image = transform.resize(image, output_shape=SIZE, order=ORDER,
                                     mode=EDGE_MODE)
            image = image * 255
            image = image.astype(np.uint8, copy=False)
            dst_path = os.path.join(example_path, "{0:s}.png".format(condition))
            imageio.imwrite(dst_path, image)
