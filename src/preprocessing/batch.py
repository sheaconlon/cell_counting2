import tensorflow as tf

def encode(patches, classes):
	return patches, tf.one_hot(classes, 2)

def batch(patches, classes, size):
    return tf.train.shuffle_batch(
        (patches, classes),
        size,
        size*20,
        size*5,
        num_threads=1,
        seed=42114,
        enqueue_many=True
    )