import numpy as np
from scipy import ndimage
from skimage import feature, morphology
import matplotlib.pyplot as plt

def count_regions(image, patch_size, patch_classifier, batch_size, min_dist):
	"""Count the number of regions in an image. Uses a classifier that can, for
		any given pixel of the image, give the probabilities of that pixel being
		inside a region, on the edge of a region, and outside all regions.

	Args:
		image (np.ndarray): The image. Must have shape (height, width,
			channels).
		patch_size (int): The side length of the patches to feed to
			patch_scorer, in pixels. Must be odd.
		patch_scorer (func): The patch scorer. Takes as argument a np.ndarray
			of shape (num_patches, patch_size, patch_size, channels) giving
			a sequence of patches of the image. Returns a np.ndarray of shape
			(num_patches, 3), where the i-th row pertains to the pixel at the
			center of the i-th patch and gives the probabilities of that pixel
			being inside a region, on the edge of a region, and outside all
			regions, respectively.
		batch_size (int): The maximum number of patches to submit to
			patch_scorer in each call.
		min_dist (int): The minimum distance between the centers of two regions,
			in pixels.

	Returns:
		(int) The number of regions in the image.
	"""
	INSIDE_CLASS = 0
	BLACK = 255

	assert patch_size > 0, "count_regions(): argument patch_size must be > 0"
	assert patch_size % 2 != 0, ("count_regions(): argument patch_size must be"
		"odd")
	assert batch_size > 0, "count_regions(): argument batch_size must be > 0"
	assert min_dist > 0, "count_regions(): argument min_dist must be > 0"

	half_size = patch_size // 2
	patches = np.zeros((batch_size, patch_size, patch_size, image.shape[2]))
	patch_num = 0
	classes = []

	print("image")
	plt.imshow(image)
	plt.show()

	image = image / 255

	def classify_batch():
		probs = patch_classifier(patches)
		classes.extend(np.argmax(probs, axis=1))

	for y in range(half_size, image.shape[0] - half_size):
		for x in range(half_size, image.shape[1] - half_size):
			xmin, xmax = x-half_size, x+half_size
			ymin, ymax = y-half_size, y+half_size
			patches[patch_num, :, :, :] = image[ymin:(ymax+1), xmin:(xmax+1), :]
			patch_num = (patch_num + 1) % batch_size
			if patch_num == 0:
				classify_batch()
	if patch_num != 0:
		classify_batch()

	inside_image = np.empty((image.shape[0] - (2*half_size),
		image.shape[1] - (2*half_size)))
	i = 0
	for y in range(half_size, image.shape[0] - half_size):
		for x in range(half_size, image.shape[1] - half_size):
			if classes[i] == INSIDE_CLASS:
				inside_image[y - half_size, x - half_size] = BLACK
			i += 1

	print("inside image")
	plt.imshow(inside_image)
	plt.show()
	dist_image = ndimage.morphology.distance_transform_edt(inside_image)
	print("dist image")
	plt.imshow(dist_image)
	plt.show()
	peak_image = feature.peak_local_max(dist_image, min_distance=min_dist,
		threshold_rel=0.5, indices=False)
	print("peak image")
	plt.imshow(peak_image)
	plt.show()
	marker_image, _ = ndimage.label(peak_image)
	print("marker image")
	plt.imshow(marker_image)
	plt.show()
	label_image = morphology.watershed(-dist_image, marker_image)
	print("label image")
	plt.imshow(label_image)
	plt.show()
	labels = np.unique(label_image)
	return labels.shape[0] - 1






	