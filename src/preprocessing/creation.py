import os, random, contextlib
from collections import Counter

from wand.image import Image

def load_images(dir, limit=None):
	"""
	Load the images contained in some directory.

	All files in the directory must be PNG images.

	Caution: This may leak resources! Proper usage of `Image` should actually
		wrap with a context manager!

	Args:
		dir (str): The path to the directory.
		limit (int): The maximum number of images to load. If not `None`, will
			load only the lexicographically first `limit` images.

	Returns:
		(list of Image): The images in the directory.
	"""
	paths = os.listdir(dir)
	if limit is not None:
		if len(paths) > limit:
			paths.sort()
			paths = paths[:limit]
	images = []
	for path in paths:
		img = Image(filename=os.path.join(dir, path))
		images.append(img)
	return images

def standardize_dimensions(images, dims, filter="gaussian", blur=1):
	"""
	Make the dimensions of the images in a list all equal.

	Args:
		images (list of Image): The images.
		dims (tuple of int): The dimensions to use.
		filter (str): The type of filter to use when rescaling. `gaussian` by
			default. See `wand.image.Image.resize`.
		blur (int): The amount of blurring to use when rescaling. `1` by
			default. See `wand.image.Image.resize`.
	"""
	for img in images:
		img.resize(dims[0], dims[1], filter, blur)

def split(images, test_p=0.1, seed=42114):
	"""
	Split a list of images into a training set and a test set, randomly.

	By choosing the same seed for splitting the images and masks of a dataset,
		one can ensure that the images and masks get outputted in the same order
		in the same subset.

	Args:
		images (list of Image): The images.
		test_p: (float): The proportion of the images which should go to the
			test set. `0.1` by default.
		seed (int): The seed to use for random number generation.

	Returns:
		(tuple of list of Image): Two lists of images, the training set and the
			test set.
	"""
	random.seed(seed)
	random.shuffle(images)
	split_index = int(len(images) * test_p)
	return images[split_index:], images[:split_index]

EXTENSION = ".png"

def save_images(images, dir):
	"""
	Save a list of images to a directory using sequential numbers as filenames.

	Args:
		images (list of Image): The images.
		dir (str): The path to the directory.
	"""
	for i, img in enumerate(images):
		img.save(filename=os.path.join(dir, str(i) + EXTENSION))
