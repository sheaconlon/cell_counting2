import tensorflow as tf

def accuracy(labels, predictions):
	one_hot_predictions = tf.one_hot(tf.argmax(predictions, axis=1), depth=tf.shape(labels)[1])
	return tf.metrics.accuracy(labels, one_hot_predictions)