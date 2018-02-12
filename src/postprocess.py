import numpy as np
from scipy import ndimage
from skimage import feature, morphology

from . import visualization

def count_regions(image, patch_size, patch_classifier, batch_size, min_dist,
                  min_diam, debug=False):
    """Count the number of regions in an image. Uses a classifier that can, for
        any given pixel of the image, give the probabilities of that pixel being
        inside a region, on the edge of a region, and outside all regions.

    Args:
        image (np.ndarray): The image. Must have shape (height, width,
            channels).
        patch_size (int): The side length of the patches to feed to
            patch_classifier, in pixels. Must be odd and at least 3.
        patch_scorer (func): The patch scorer. Takes as argument a np.ndarray
            of shape (num_patches, patch_size, patch_size, channels) giving
            a sequence of patches of the image. Returns a np.ndarray of shape
            (num_patches, 3), where the i-th row pertains to the pixel at the
            center of the i-th patch and gives the probabilities of that pixel
            being inside a region, on the edge of a region, and outside all
            regions, respectively.
        batch_size (int): The ideal number of patches to submit to
            patch_scorer in each call.
        min_dist (int): The minimum distance between the centers of two regions,
            in pixels.
        min_diam (float): The minimum diameter of a region, in pixels.
        debug (bool): Whether to plot each intermediate step for debugging
            purposes. False by default.

    Returns:
        (int) The number of regions in the image.
    """
    RGB_MAX = 255
    DEBUG_PLOT_DIMS = (3, 3)
    INSIDE_CLASS = 0

    assert patch_size >= 3, "argument patch_size must be >= 3"
    assert patch_size % 2 != 0, "argument patch_size must be odd"
    assert batch_size > 0, "argument batch_size must be > 0"
    assert min_dist > 0, "argument min_dist must be > 0"

    half_size = patch_size // 2
    patch_batch = np.zeros((batch_size, patch_size, patch_size, image.shape[2]))
    patch_num = 0
    classes = []

    image = image / RGB_MAX
    if debug:
        visualization.plot_images(image[np.newaxis, ...], 1, DEBUG_PLOT_DIMS,
                                  "original")

    def classify_batch(n):
        probs = patch_classifier(patch_batch[:n, ...])
        classes.extend(np.argmax(probs, axis=1))

    for y in range(half_size, image.shape[0] - half_size):
        for x in range(half_size, image.shape[1] - half_size):
            xmin, xmax = x-half_size, x+half_size
            ymin, ymax = y-half_size, y+half_size
            patch_batch[patch_num, ...] = image[ymin:(ymax+1), xmin:(xmax+1), :]
            patch_num = (patch_num + 1) % batch_size
            if patch_num == 0:
                classify_batch(batch_size)
    if patch_num != 0:
        classify_batch(patch_num)

    inside_height = image.shape[0] - (2*half_size)
    inside_width = image.shape[1] - (2*half_size)
    inside_image = np.full((inside_height, inside_width), False)
    patch_num = 0
    for y in range(inside_height):
        for x in range(inside_width):
            if classes[patch_num] == INSIDE_CLASS:
                inside_image[y, x] = True
            patch_num += 1
    if debug:
        visualization.plot_images(inside_image[np.newaxis, ...], 1,
                                  DEBUG_PLOT_DIMS, "inside")

    dist_image = ndimage.morphology.distance_transform_edt(inside_image)
    if debug:
        visualization.plot_images(dist_image[np.newaxis, ...], 1,
                                  DEBUG_PLOT_DIMS, "distance")

    peak_image = feature.corner_peaks(dist_image, min_distance=min_dist,
        threshold_abs=min_diam/2, indices=False)
    peak_image = peak_image.astype(int)
    if debug:
        visualization.plot_images(peak_image[np.newaxis, ...], 1,
                                  DEBUG_PLOT_DIMS, "peak")

    marker_image, _ = ndimage.label(peak_image)
    if debug:
        visualization.plot_images(marker_image[np.newaxis, ...], 1,
                                  DEBUG_PLOT_DIMS, "marker")

    label_image = morphology.watershed(-dist_image, marker_image,
                                       mask=inside_image)
    if debug:
        visualization.plot_images(label_image[np.newaxis, ...]
                                  , 1,
                                  DEBUG_PLOT_DIMS, "label")

    labels = np.unique(label_image)
    return labels.shape[0] - 1
