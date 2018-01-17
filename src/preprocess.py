import math

import tensorflow as tf

import numpy as np
from sklearn.feature_extraction import image as img

def smdm_normalize(images, window, padding):
	"""
	Normalizes an image using the "divide by global median, subtract local mean"
		approach.

	Normalizes each channel independently.

	Args:
		images (np.ndarray): An array of images. Assumed to have shape
			(batch_size, height, width, channels).
		window (int): The window size that should be used for the "subtract
			local mean" step. Must be odd.
		padding (str): The type of padding to use when computing local means.
			Must be one of "CONSTANT", "REFLECT", or "SYMMETRIC".

	Returns:
		(np.ndarray) A normalized array of images.

	Throws:
		ValueError: The window size was even.
	"""
	if window % 2 == 0:
		raise ValueError("called smdm_normalize with even-sized window")

	images = tf.constant(images)
	images = tf.cast(images, tf.float32)
	batch_size = tf.shape(images)[0]
	height = tf.shape(images)[1]
	width = tf.shape(images)[2]
	channels = tf.shape(images)[3]

	spatial_last = tf.transpose(images, (0, 3, 1, 2))
	spatial_last_and_flat = tf.reshape(spatial_last, (batch_size, channels, -1))
	n = tf.multiply(height, width)
	k = tf.to_int32(tf.divide(n, 2)) + 1
	top_k = tf.nn.top_k(spatial_last_and_flat, k)[0]
	medians_spatial_last_and_flat = tf.cond(
		tf.equal(tf.mod(n, 2), 0),
		lambda: tf.reduce_mean(top_k[:, :, k - 2: k], -1, keep_dims=True),
		lambda: top_k[:, :, k - 1]
	)
	medians_spatial_last_and_flat = tf.add(
		medians_spatial_last_and_flat,
		tf.fill(tf.shape(medians_spatial_last_and_flat), tf.constant(1e-8))
	)
	medians_spatial_last = tf.expand_dims(medians_spatial_last_and_flat, 3)
	medians = tf.transpose(medians_spatial_last, (0, 2, 3, 1))
	images = tf.divide(images, medians)

	padding_amount = int((window - 1) / 2)
	padding_amounts = (
		(0, 0),
		(padding_amount, padding_amount),
		(padding_amount, padding_amount),
		(0, 0)
	)
	images_padded = tf.pad(images, padding_amounts, padding)
	local_means = tf.nn.pool(images_padded, (window, window), "AVG", "VALID")
	images = tf.subtract(images, local_means)

	with tf.Session() as sess:
		images = sess.run(images)
	return images

def one_hot_encode(labels, num_classes):
	"""
	One-hot encodes labels.

	Args:
		labels (np.ndarray): An array of labels. Assumed to have shape
			(batch_size, 1).
		num_classes (int): The number of classes.

	Returns:
		(np.ndarray) An array of one-hot encoded labels with shape
			(batch_size, num_classes)
	"""
	with tf.Session() as sess:
		return sess.run(tf.one_hot(labels, num_classes))

def extract_patches(image, size, max_patches=None):
	"""
	Extracts square patches of an image.

	Args:
		image (np.ndarray): The image. Must have shape (height, width, 3).
		size (int): The side length (in pixels) of the patches.
		max_patches (int): The maximum number of patches to extract. If omitted,
			all possible patches are extracted. The actual number of patches
			extracted may be somewhat less than this but will not exceed this.

	Returns:
		(np.ndarray): Patches of the image. Has shape (num_patches, size, size,
			3).
	"""
	if max_patches is not None:
		possible_patches = (image.shape[0] - size + 1) * \
			(image.shape[1] - size + 1)
		frac_max_patches = max_patches / possible_patches
		stride = math.ceil(math.sqrt(1 / frac_max_patches))
	else:
		stride = 1
	patches_by_channel = []
	for channel_index in range(3):
		with tf.Session() as sess:
			channel = image[:, :, (channel_index,)]
			images = tf.constant(channel)
			images = tf.expand_dims(images, axis=0)
			patches = tf.extract_image_patches(images, (1, size, size, 1),
				(1, stride, stride, 1), (1, 1, 1, 1), "VALID")
			patches = tf.reshape(patches, (-1, size, size, 1))
			patches_by_channel.append(sess.run(patches))
	return np.concatenate(patches_by_channel, axis=3)
