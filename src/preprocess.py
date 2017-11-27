import tensorflow as tf
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

def extract_patches(image, size, max_patches=None, seed=None):
	"""
	Extracts square patches from an image.

	Args:
		image (np.ndarray): An image. Should have shape (height, width) or
			(height, width, num_channels).
		size (int): The side length of a single patch.
		max_patches (int): The maximum number of patches to extract.
		seed (int): The seed for random selection of patches when max_patches is
			used.

	Returns:
		(np.ndarray): Patches of the the image. Has shape
			(num_patches, size, size) or
			(num_patches, size, size, num_channels), depending on the shape
			of the image.
	"""
	return img.extract_patches_2d(image, (size, size), max_patches=max_patches,
		random_state=seed)
