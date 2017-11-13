import os

import tensorflow as tf
import numpy as np

EXTENSION = ".png"
IMAGES_DIRECTORY = "images"
MASKS_DIRECTORY = "masks"
IMAGE_CHANNELS = 3
MASKS_CHANNELS = 1

def load_images_and_masks(path, names):
	"""
	Loads a dataset consisting of some images and their masks.

	Both the images and the masks must be in PNG format. All images and all
		masks must have the same dimensions. Images must have 3 channels and
		masks must have 1 channel. Images will be found in the `images`
		subdirectory and masks will be found in the `masks` subdirectory. All
		files must have extension `.png`.

	Args:
		path (str): The directory containing the images and masks.
		names (list of str): The names of the images (and masks) to load.

	Returns:
		(tuple of tf.Tensor): The images and masks. The first tensor is the
			images and has shape `[len(names), height_of_images,
			width_of_images, 3]`. The second tensor is the masks and has shape
			`[len(names), height_of_images, width_of_images, 1]`.

	Raises:
		OSError: Some image or mask file did not exist
	"""
	images_path = os.path.join(path, IMAGES_DIRECTORY)
	images = load_images(images_path, names, IMAGE_CHANNELS)
	masks_path = os.path.join(path, MASKS_DIRECTORY)
	masks = load_images(masks_path, names, MASKS_CHANNELS)
	return images, masks

def load_images(path, names, n_channels):
	"""
	Loads a dataset consisting of some images.

	The images must be in PNG format. All images must have the same dimensions
		and same number of channels. All images must have extension `.png`.

	Args:
		path (str): The directory containing the images.
		names (list of str): The names of the images to load.
		n_channels (int): The number of channels in the images.

	Returns:
		(tf.Tensor): The images. Has shape `[len(names), height_of_images,
			width_of_images, n_channels]`. 

	Raises:
		OSError: Some image file did not exist
	"""
	images = []
	for name in names:
		image_path = os.path.join(path, name + EXTENSION)
		with open(image_path, "rb") as image_file:
			image = tf.image.decode_image(image_file.read(), n_channels)
		image = tf.expand_dims(image, 0)
		images.append(image)
	return tf.concat(images, 0)

def load_csv(path):
	"""
	Loads a dataset from some CSV file.

	Args:
		path (str): The path to the file.

	Returns:
		(np.ndarray of X): An array representing the CSV file.
	"""
	return np.loadtxt(path, ",")
