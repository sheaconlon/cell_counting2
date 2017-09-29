import os

import tensorflow as tf

def load(path, n_channels, name="unnamed_image_decode"):
	"""
	Loads the files contained in some directory as images.

	The files must be in PNG or JPEG format.

	Args:
		path: A string giving the path of the directory.
		n_channels: An integer giving the number of channels in the images.
		name: The base name that should be used to name TensorFlow nodes.

	Returns:
		A list of `Tensor`s of type `uint8`. Each `Tensor` gives the data for one image and has shape `[height, width, num_channels]`.

	Raises:
		ValueError: Some image in `path` did not have `n_channels` channels or some image in `path` was of type not in {PNG, JPEG}.
	"""
	images = []
	for i, filename in enumerate(os.listdir(path)):
		full_path = os.path.join(path, filename)
		file = open(full_path, "rb")
		image = tf.image.decode_image(file.read(), n_channels, name + "_" + str(i))
		file.close()
		images.append(image)
	return images
