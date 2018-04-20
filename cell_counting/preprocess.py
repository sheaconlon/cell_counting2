import math, random

import tensorflow as tf
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa

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

def divide_median_normalize(images):
    """Normalize some images by dividing by their median values.

    The normalization is performed independently for each channel of each image.

    Args:
        images (np.ndarray): The images. Must have shape (num_images, height,
            width, num_channels).

    Returns:
        (np.ndarray): The normalized images.
    """
    assert images.ndim == 4, "parameter images must have 4 dimensions"

    medians = np.median(images, axis=(1, 2))
    if np.any(medians == 0):
        raise ValueError("zero median encountered; cannot divide by zero")
    return images / medians[:, np.newaxis, np.newaxis, :]

def subtract_mean_normalize(images):
    """Normalize some images by subtracting their mean values.

    The normalization is performed independently for each channel of each image.

    Args:
        images (np.ndarray): The images. Must have shape (num_images, height,
            width, num_channels).

    Returns:
        (np.ndarray): The normalized images.
    """
    assert images.ndim == 4, "parameter images must have 4 dimensions"

    means = np.mean(images, axis=(1, 2))
    return images - means[:, np.newaxis, np.newaxis, :]

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

def extract_patches_generator(image, class_image, size, max_patches=float("inf"),
    class_dist=None):
    """
    Extract square patches of an image and their corresponding classes.

    Uses the "VALID" padding method, meaning that no patches centered near the
        edge of the image are extracted.

    Args:
        image (np.ndarray): The image. Must have shape
            (height, width, channels).
        class_image (np.ndarray): The class image. Must have shape
            (height, width) and type int. classes[y, x] gives the class of the
            patch centered at (x, y) in the image.
        size (int): The desired side length (in pixels) of the patches. Must be
            odd.
        max_patches (int): The maximum number of patches. Up to this many
            patches will be returned. Fewer will be returned if necessary to
            produce the desired class_dist. If omitted or None, the maximum is
            infinity.
        class_dist (dict(int, float)): The desired class distribution of the
            patches. class_dist[cls] is the fraction of the patches which should
            be of cls. If omitted or None, a uniform distribution is assumed.
            Note that the resulting class distribution will not be exactly
            class_dist.

    Yields:
        (tuple): A tuple consisting of a patch (a `numpy.ndarray` with shape
            (``size``, ``size``, ``image.shape[2]``)) and its class.

    Throws:
        ValueError if there are not enough patches of some class to fulfill
            the request.
    """
    assert image.ndim == 3, "image does not have 3 dims"
    assert len(class_image.shape) == 2, "class image does not have 2 dims"
    assert image.shape[0] == class_image.shape[0] \
        and image.shape[1] == class_image.shape[1], \
        "image and class image do not have matching height and width"
    assert size < image.shape[0] and size < image.shape[1], \
        "patch size too large for image"
    assert size % 2 == 1, "size not odd"
    assert max_patches is None or max_patches > 0, \
        "number of patches is not positive"
    assert class_dist is None or ( \
        math.isclose(sum(class_dist.values()), 1) \
        and all(x > 0 for x in class_dist.values())), \
        "class distribution is not a proper distribution"

    # Generate, for each class, a list of locations whose surrounding patch
    # is of that class.
    class_locations = {}
    half_size = size // 2
    for y in range(half_size, class_image.shape[0] - half_size):
        for x in range(half_size, class_image.shape[1] - half_size):
            class_ = class_image[y, x]
            if class_ not in class_locations:
                class_locations[class_] = []
            class_locations[class_].append((x, y))

    # The default class distribution is uniform.
    if class_dist is None:
        num_classes = len(class_locations)
        class_dist = {class_: 1 / num_classes for class_ in class_locations}

    # The default number of patches is as many as possible while still
    # providing the desired class distribution.
    limiting_ratio, limiting_class = float("-inf"), None
    num_patches = (class_image.shape[0] - 2*half_size) \
        * (class_image.shape[1] - 2*half_size)
    for class_ in class_dist:
        actual_frac = len(class_locations[class_]) / num_patches
        desired_frac = class_dist[class_]
        ratio = desired_frac / actual_frac
        if ratio > limiting_ratio:
            limiting_ratio, limiting_class = ratio, class_
    limiting_num_patches = len(class_locations[limiting_class])
    n = math.floor(limiting_num_patches / class_dist[limiting_class])
    n = min(max_patches, n)

    # Extract the patches.
    for class_ in class_dist:
        # In the following line, we choose to extract too many patches of this
        # class rather than too few.
        class_n = math.ceil(n * class_dist[class_])
        if class_n > len(class_locations[class_]):
            raise ValueError(
                "there are not enough patches of class {0:d}".format(class_))
        for location in random.sample(class_locations[class_], class_n):
            x, y = location
            xmin, xmax = x-half_size, x+half_size
            ymin, ymax = y-half_size, y+half_size
            patch = image[ymin:(ymax+1), xmin:(xmax+1), :]
            yield (patch, class_)

def extract_patches(image, class_image, size, max_patches=float("inf"),
    class_dist=None):
    """
    Extract square patches of an image and their corresponding classes.

    See ``extract_patches_generator``.

    Args:
        image (np.ndarray): See ``extract_patches_generator``.
        class_image (np.ndarray): See ``extract_patches_generator``.
        size (int): See ``extract_patches_generator``.
        max_patches (int): See ``extract_patches_generator``.
        class_dist (dict(int, float)): See ``extract_patches_generator``.

    Returns:
        (tuple(np.ndarray, np.ndarray)): Patches of the image (shape
            (n, size, size, 3)) and their corresponding classes
            (shape (n)).

    Throws:
        See ``extract_patches_generator``.
    """
    patches, classes = [], []
    for patch, class_ in extract_patches_generator(image, class_image, size,
                                                   max_patches, class_dist):
        patches.append(patch)
        classes.append(class_)

    # Shuffle the patches.
    shuffle_order = list(range(patches.shape[0]))
    random.shuffle(shuffle_order)
    patches_shuffled, classes_shuffled = [], []
    for i in shuffle_order:
        patches_shuffled.append(patches[i])
        classes_shuffled.append(classes[i])
    # In the following lines, since we may have extracted too many patches,
    # we take only the first n.
    patches_shuffled = np.stack(patches_shuffled, axis=0)[:max_patches, ...]
    classes_shuffled = np.stack(classes_shuffled, axis=0)[:max_patches, ...]
    return patches_shuffled, classes_shuffled


def _var_of_vars(size, images, samples):
    BATCH_SIZE = 1000

    num_images, height, width, num_channels = images.shape
    size = int(size)
    if size % 2 == 0:
        size -= 1
    half_size = size // 2
    patches_per_row = width - 2*half_size
    patches_per_image = (height - 2*half_size) * patches_per_row
    num_patches = num_images * patches_per_image
    chosen_patch_nums = np.random.choice(num_patches, samples)
    batch_patch_num = 0
    batch = np.empty((BATCH_SIZE, size, size, num_channels))
    vars = []
    for i, patch_num in enumerate(chosen_patch_nums):
        image_num = patch_num // patches_per_image
        patch_num = patch_num % patches_per_image
        y = patch_num // patches_per_row + half_size
        x = patch_num % patches_per_row + half_size
        xmin, xmax = x-half_size, x+half_size
        ymin, ymax = y-half_size, y+half_size
        patch = images[image_num, ymin:(ymax+1), xmin:(xmax+1), :]
        batch[batch_patch_num, :, :, :] = patch
        batch_patch_num += 1
        if batch_patch_num == BATCH_SIZE:
            vars.extend(np.var(batch, axis=(1, 2, 3)))
            batch_patch_num = 0
    vars.extend(np.var(batch, axis=(1, 2, 3)))
    var_of_vars = np.var(vars, ddof=1)
    return size, float(var_of_vars)


def patch_variability_curve(images, min_size, max_size, num_sizes, samples):
    """Calculate the patch variability curve of some images.

    The patch variability curve relates patch size to the variance of the
        variances of the patches of that size. Variances are taken over all
        the positions and all the channels of a patch. Patch sizes are sampled
        logarithmically from the range [min_size, max_size]. min_size and
        max_size are included.

    See `extract_patches` for further information about patches.

    Args:
        images (np.ndarray): The images. Must have shape ``(num_images, height,
            width, num_channels)``.
        min_size (int): The smallest patch size to sample. Must be odd.
        max_size (int): The largest patch size to sample. Must be odd.
        num_sizes (int): The number of patch sizes to sample.
        samples (int): The number of patches to sample for the variance
            calculation for each patch size.

    Returns:
        tuple(numpy.ndarray, numpy.ndarray): The first array lists the
        sampled patch sizes, in ascending order. The second array contains the
        corresponding variances of variances. Both are of shape ``(num_sizes)``.
    """
    assert images.ndim == 4, "parameter images must have 4 dimensions"
    assert min_size >= 3, "parameter min_size must be at least 3"
    assert min_size % 2 == 1, "parameter min_size must be odd"
    assert max_size  <= min(images.shape[1:2]), "parameter max_size must be " \
                                                "smaller than images.shape[1]" \
                                                "and images.shape[2]"
    assert min_size % 2 == 1, "parameter max_size must be odd"
    assert min_size <= max_size, "parameter min_size must be <= parameter " \
                                 "max_size"
    assert num_sizes >= 2, "parameter num_sizes must be >= 2"
    assert samples >= 1, "parameter samples must be >= 1"

    sizes, var_of_vars = np.empty(num_sizes), np.empty(num_sizes)
    intended_sizes = np.geomspace(min_size, max_size, num=num_sizes)
    for i, intended_size in enumerate(intended_sizes):
        sizes[i], var_of_vars[i] = _var_of_vars(intended_size, images, samples)
    return sizes, var_of_vars

