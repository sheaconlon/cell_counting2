import tensorflow as tf

def smdm_normalize(images, window, padding, name="unnamed_smdm_normalize"):
	"""
	Normalizes an image using the "divide by global median, subtract local mean" approach.

	Args:
		images: An batch of images as a `Tensor`. First dimension is batch. Next two dimensions are y and x coordinates, respectively.
				Last dimension is channel.
		window: The window size that should be used for the "subtract local mean" phase. Must be odd.
		padding: The type of padding to use when computing local means. Must be one of `"CONSTANT"`, `"REFLECT"`, or `"SYMMETRIC"`.
		name: The base name that should be used to name TensorFlow nodes.

	Returns:
		`image`, a `Tensor` of `float`s with its pixel values within each channel normalized.

	Throws:
		ValueError: The window size was even.
	"""
	MEDIAN_JITTER = tf.constant(1e-8)
	
	if window % 2 == 0:
		raise ValueError("attempted to smdm_normalize() with even-sized window")

	images = tf.cast(images, tf.float32)
	batch_size, height, width, channels = tf.shape(images)[0], tf.shape(images)[1], tf.shape(images)[2], tf.shape(images)[3]

	spatial_last = tf.transpose(images, (0, 3, 1, 2))
	spatial_last_and_flat = tf.reshape(spatial_last, (batch_size, channels, -1))
	n = tf.multiply(height, width)
	k = tf.to_int32(tf.divide(n, 2)) + 1
	top_k = tf.nn.top_k(spatial_last_and_flat, k, name=name + "_top_half_of_images")[0]
	medians_spatial_last_and_flat = tf.cond(
		tf.equal(tf.mod(n, 2), 0),
		lambda: tf.reduce_mean(top_k[:, :, k - 2: k], -1, keep_dims=True),
		lambda: top_k[:, :, k - 1]
	)
	medians_spatial_last_and_flat = tf.add(
		medians_spatial_last_and_flat,
		tf.fill(tf.shape(medians_spatial_last_and_flat), MEDIAN_JITTER)
	)
	medians_spatial_last = tf.expand_dims(medians_spatial_last_and_flat, 3)
	medians = tf.transpose(medians_spatial_last, (0, 2, 3, 1))
	images = tf.divide(images, medians, name=name + "_divide_images_by_medians")

	padding_amount = int((window - 1) / 2)
	padding_amounts = ((0, 0), (padding_amount, padding_amount), (padding_amount, padding_amount), (0, 0))
	images_padded = tf.pad(images, padding_amounts, padding)
	local_means = tf.nn.pool(images_padded, (window, window), "AVG", "VALID", name=name + "_local_means_of_images")
	images = tf.subtract(images, local_means, name=name + "_subtract_local_means_from_images")

	return images