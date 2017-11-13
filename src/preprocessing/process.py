import tensorflow as tf
import scipy

PADDING_TYPE = "VALID"

def _class(mask_patch):
    middle_y = tf.floordiv(tf.shape(mask_patch)[0], tf.constant(2))
    middle_x = tf.floordiv(tf.shape(mask_patch)[1], tf.constant(2))
    return tf.cond(
        tf.less(mask_patch[middle_y, middle_x, 0], 10),
        lambda: tf.constant(1, dtype=tf.uint8),
        lambda: tf.constant(0, dtype=tf.uint8)
    )

def patchify(images, masks, window_side, window_stride):
    """
    Patchify a dataset consisting of some images and their masks.

    Args:
        images (tf.Tensor): The images.
        masks (tf.Tensor): The masks.
        window_side (int): The side length of the patch window, in pixels.
        window_stride (int): The length of the patch window stride, in pixels.

    Returns:
        (tuple of tf.Tensor): Image patches and their classes. First tensor is
            the image patches and has shape `[n_patches, window_side,
            window_side, n_channels]`. Second tensor is the classes and has
            shape `[n_patches]`.
    """
    sizes = (1, window_side, window_side, 1)
    strides = (1, window_stride, window_stride, 1)
    rates = (1, 1, 1, 1)

    patches = tf.extract_image_patches(images, sizes, strides, rates,
        PADDING_TYPE)
    patches = tf.reshape(patches, (-1, window_side, window_side, 3))
    classes = tf.extract_image_patches(masks, sizes, strides, rates,
        PADDING_TYPE)
    classes = tf.reshape(classes, (-1, window_side, window_side, 1))
    classes = tf.map_fn(_class, classes)

    return patches, classes

def one_hot_encode(classes):
    return tf.one_hot(classes, 2)
