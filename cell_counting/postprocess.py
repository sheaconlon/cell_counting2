from . import visualization

import numpy as np
from scipy import ndimage, stats
from skimage import feature, morphology

def count_regions(image, patch_size, patch_classifier, batch_size, min_dist,
                  min_diam, sampling_interval=1, debug=False):
    """Count the number of regions in an image.

    Uses a classifier that can, for any given pixel of the image, give the
        probabilities of that pixel being inside a region, on the edge of a
        region, and outside all regions. Treats the edges of the image
        accoring to the "valid" strategy. Note that the patches will not be
        shuffled and will therefore exhibit spatial correlation.


    Args:
        image (np.ndarray): The image. Must have shape (height, width,
            channels).
        patch_size (int): The side length of the patches to feed to
            patch_classifier, in pixels. Must be odd and at least 3.
        patch_classifier (func): The patch scorer. Takes as argument a
            np.ndarray of shape (num_patches, patch_size, patch_size,
            channels) giving a sequence of patches of the image. Returns a
            np.ndarray of shape (num_patches, 3), where the i-th row pertains to
            the pixel at the center of the i-th patch and gives the
            probabilities of that pixel being inside a region, on the edge of a
            region, and outside all regions, respectively.
        batch_size (int): The ideal number of patches to submit to
            patch_scorer in each call.
        min_dist (int): The minimum distance between the centers of two regions,
            in pixels.
        min_diam (float): The minimum diameter of a region, in pixels.
        sampling_interval (int): In every square of the image with side length
            sampling_interval pixels, a single pixel will be chosen for
            classification. The default value is 1, which causes every pixel to
            be classified.
        debug (bool): Whether to plot each intermediate step for debugging
            purposes. False by default.

    Returns:
        (int, tuple(int, dict)) If ``debug`` is ``False``, the number of
        regions in the image. If ``debug`` is ``True``, a tuple whose first
        element is the number of regions in the image and whose second
        element is a dictionary. This dictionary contains the original image
        under the key ``"original"``, the inside mask produced by the classifier
        under the key ``"inside"``, the distance-transformed image under the key
        ``"distance"``, the peak-identified image under the key ``"peak"``,
        the marker-labeled image under the key ``"marker"``, and the final label
        image under the key ``"label"``.
    """
    RGB_MAX = 255
    DEBUG_PLOT_DIMS = (8, 8)
    INSIDE_CLASS = 0

    assert patch_size >= 3, "argument patch_size must be >= 3"
    assert patch_size % 2 != 0, "argument patch_size must be odd"
    assert batch_size > 0, "argument batch_size must be > 0"
    assert min_dist > 0, "argument min_dist must be > 0"
    assert sampling_interval >= 1, "argument sampling_interval must be >= 1"

    half_size = patch_size // 2
    patch_batch = np.zeros((batch_size, patch_size, patch_size, image.shape[2]))
    patch_num = 0
    classes = []

    def classify_batch(n):
        probs = patch_classifier(patch_batch[:n, ...])
        classes.extend(np.argmax(probs, axis=1))

    for y in range(half_size, image.shape[0] - half_size, sampling_interval):
        for x in range(half_size, image.shape[1] - half_size,
                       sampling_interval):
            xmin, xmax = x-half_size, x+half_size
            ymin, ymax = y-half_size, y+half_size
            patch_batch[patch_num, ...] = image[ymin:(ymax+1), xmin:(xmax+1), :]
            patch_num = (patch_num + 1) % batch_size
            if patch_num == 0:
                classify_batch(batch_size)
    if patch_num != 0:
        classify_batch(patch_num)

    inside_height = len(range(half_size, image.shape[0] - half_size,
                              sampling_interval))
    inside_width = len(range(half_size, image.shape[1] - half_size,
                             sampling_interval))
    inside_image = np.full((inside_height, inside_width), False)
    patch_num = 0
    for y in range(inside_height):
        for x in range(inside_width):
            if classes[patch_num] == INSIDE_CLASS:
                inside_image[y, x] = True
            patch_num += 1

    dist_image = ndimage.morphology.distance_transform_edt(inside_image)

    peak_image = feature.corner_peaks(dist_image, min_distance=min_dist,
        threshold_abs=min_diam/2, indices=False)
    peak_image = peak_image.astype(int)

    marker_image, _ = ndimage.label(peak_image)

    label_image = morphology.watershed(-dist_image, marker_image,
                                       mask=inside_image)

    labels = np.unique(label_image)
    count = labels.shape[0] - 1
    if debug:
        return (count, {"original": image, "inside": inside_image,
                        "distance": dist_image, "peak": peak_image,
                        "marker": marker_image, "label": label_image})
    else:
        return count

def confidence_cutoff_analysis(probs, classes, cutoff_samples=1000):
    """Calculate a confidence cutoff analysis.

    A confidence cutoff analysis trials a number of confidence cutoffs for
        determining whether to accept predictions. For each cutoff, the accuracy
        level that would've been achieved with that cutoff and the proportion of
        predictions that would've been accepted with that cutoff are calculated.
        The confidence of a prediction is calculated as the entropy of its
        probability distribution over classes.

    Args:
        probs (numpy.ndarray): The predicted class probabilities. An array with
            shape ``(num_examples, num_classes)``. May be unnormalized.
        classes (Iterable[int]): The actual classes of the examples. Must have
            length ``num_examples``.

    Returns:
        (tuple[numpy.ndarray]): Three arrays. The first contains the confidence
            level cutoffs trialed. The second contains the accuracy levels
            that would've been achieved with those confidence level cutoffs. The
            third contains the proportions of predictions that would've met
            those confidence level cutoffs. All have shape ``(num_levels)``.
    """
    num_examples, _ = probs.shape
    accs = []
    props = []
    pred_classes = np.argmax(probs, axis=1)
    confs = [stats.entropy(probs[i, ...]) for i in range(num_examples)]
    cutoffs = np.linspace(min(confs), max(confs), num=cutoff_samples)
    for cutoff in cutoffs:
        met_and_correct = 0
        met = 0
        for i, cls in zip(range(num_examples), classes):
            if confs[i] <= cutoff:
                met += 1
                if pred_classes[i] == cls:
                    met_and_correct += 1
        accs.append(met_and_correct / met)
        props.append(met / num_examples)
    return cutoffs, np.array(accs), np.array(props)
