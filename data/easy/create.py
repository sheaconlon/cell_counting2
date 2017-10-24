import sys

sys.path.append("../..")

from src.preprocessing.creation import *

LIMIT = 100
DIMS = (2448, 2448)
FILTER = "gaussian"
BLUR = 1
TEST_P = 0.1
SEED = 42114

print("loading images")
images = load_images("raw/images", LIMIT)
print("standardizing dimensions of images")
standardize_dimensions(images, DIMS, filter=FILTER, blur=BLUR)
print("splitting images")
train_images, test_images = split(images, test_p=TEST_P, seed=SEED)
print("saving training images")
save_images(train_images, "train/images")
print("saving test images")
save_images(test_images, "test/images")

print("loading labels")
labels = load_excel("raw/Plates.xlsx", "A", "C", LIMIT)
print("splitting labels")
train_labels, test_labels = split(labels, test_p=TEST_P, seed=SEED)
print("saving training labels")
save_csv(train_labels, "train/labels.csv")
print("saving test labels")
save_csv(test_labels, "test/labels.csv")
