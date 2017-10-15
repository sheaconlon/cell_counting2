import tensorflow as tf

QUEUE_CAPCITY_FACTOR = 100

def input_fn(patches, targets, batch_size, epochs=None):
	with tf.Session() as data_sess:
		patches = data_sess.run(patches)
		targets = data_sess.run(targets)
	underlying = tf.estimator.inputs.numpy_input_fn(
    	{"patches":patches},
    	y=targets,
    	batch_size=batch_size,
    	num_epochs=epochs,
    	shuffle=True,
    	queue_capacity=QUEUE_CAPCITY_FACTOR*batch_size,
    	num_threads=1
	)
	def fn():
		input_dict, targets = underlying()
		return input_dict["patches"], targets
	return fn