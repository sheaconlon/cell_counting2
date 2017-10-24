import os, multiprocessing, random, json

import numpy as np
import scipy
import scipy.ndimage as ndimage

SEED = 42114
TEST_P = 0.1
SIZE = (128, 128)

def process(example):
	segment_path = os.path.join("raw", example["Segment Relative Path"])
	mask_path = os.path.join("raw", example["Binary Segment Relative Path"])
	label = example["data"]["segment_type"]["data"]
	segment = ndimage.imread(segment_path, flatten=False)
	segment = scipy.misc.imresize(segment, SIZE, interp="bicubic")
	mask = ndimage.imread(mask_path, flatten=False)
	mask = scipy.misc.imresize(mask, SIZE, interp="bicubic")
	segment = segment - mask
	processed_example = (segment_path, segment, label)
	return processed_example

examples_file = open("raw/enumeration_segments.json")
examples = json.load(examples_file)
n_proc = len(os.sched_getaffinity(0))
with multiprocessing.Pool(n_proc) as pool:
	processed_examples = pool.map(process, examples.values())
processed_examples.sort()
random.seed(SEED)
random.shuffle(processed_examples)
n_test = int(TEST_P * len(processed_examples))
train_examples = processed_examples[n_test:]
test_examples = processed_examples[:n_test]
train_images, train_labels = zip(*[(ex[1], ex[2]) for ex in train_examples])
test_images, test_labels = zip(*[(ex[1], ex[2]) for ex in test_examples])
train_images = np.stack(train_images)
train_labels = np.stack(train_labels)
test_images = np.stack(test_images)
test_labels = np.stack(test_labels)
np.save("train_images.npy", train_images)
np.save("train_labels.npy", train_labels)
np.save("test_images.npy", test_images)
np.save("test_labels.npy", test_labels)