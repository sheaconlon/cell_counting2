import tensorflow as tf

def accuracy(labels, predictions):
	argmax_labels = tf.argmax(labels, axis=1)
	argmax_predictions = tf.argmax(predictions, axis=1)
	return tf.metrics.accuracy(argmax_labels, argmax_predictions)

def false_negative_rate(labels, predictions):
	argmax_labels = tf.argmax(labels, axis=1)
	argmax_predictions = tf.argmax(predictions, axis=1)
	return tf.metrics.false_negatives(argmax_labels, argmax_predictions)

def false_positive_rate(labels, predictions):
	argmax_labels = tf.argmax(labels, axis=1)
	argmax_predictions = tf.argmax(predictions, axis=1)
	return tf.metrics.false_positives(argmax_labels, argmax_predictions)

def true_positive_rate(labels, predictions):
	argmax_labels = tf.argmax(labels, axis=1)
	argmax_predictions = tf.argmax(predictions, axis=1)
	return tf.metrics.true_positives(argmax_labels, argmax_predictions)

def true_negative_rate(labels, predictions):
	argmax_labels_inverse = tf.argmax(1 - labels, axis=1)
	argmax_predictions_inverse = tf.argmax(1 - predictions, axis=1)
	return tf.metrics.true_positives(argmax_labels_inverse, argmax_predictions_inverse)

def roc_auc(labels, predictions):
	argmax_labels = tf.argmax(labels, axis=1)
	argmax_predictions = tf.argmax(predictions, axis=1)
	return tf.metrics.auc(argmax_labels, argmax_predictions)

