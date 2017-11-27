import tensorflow as tf

def make_class_difference_loss(class_fn=lambda x: x, exp=1):
	"""Make a class-difference loss function.

	The cvlass difference loss is this:
	\sum_{(c,y)} \sum_{i=0}^7 y_i|class_fn(i) - class_fn(c)|^exp

	c is the correct class for an example
	y is the softmax of the model's output for an example (a vector)

	Args:
		class_fn (func(tf.Tensor) -> tf.Tensor): A function that takes class
			numbers and returns numbers. Applied to the class numbers before
			they are used in the calculation. Default is the identity function.
		exp (float): The exponent on the class difference. Default is 1.
	"""
	def class_difference_loss(actual, pred):
		pred_cls = tf.argmax(pred, axis=1)
		actual_cls = tf.argmax(actual, axis=1)

		pred_cls = class_fn(pred_cls)
		actual_cls = class_fn(actual_cls)

		diffs = tf.subtract(pred_cls, actual_cls)
		diffs = tf.abs(diffs)
		diffs = tf.pow(diffs, exp)
		return tf.reduce_sum(diffs, name="loss")
	return class_difference_loss