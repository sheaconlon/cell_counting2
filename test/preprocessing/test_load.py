import unittest

import tensorflow as tf

from src.preprocessing import load

class LoadTestCase(unittest.TestCase):
	"""Tests for load.py."""

	PATH = "test/preprocessing/load_test_load"
	CHANNELS = 3
	SIZE = (458, 370)
	TEST_1_LOCATION = (156, 191)
	TEST_1_COLOR = (251, 41, 26)
	TEST_2_LOCATION = (10, 10)
	TEST_2_COLOR = (255, 255, 255)
	COLOR_DELTA = 5
	N_IMAGES = 2

	def test_load_load(self):
		"""Does load() correctly load example PNG and JPEG files?"""
		sess = tf.Session()
		images = load.load(LoadTestCase.PATH, LoadTestCase.CHANNELS)
		self.assertEqual(sess.run(tf.rank(images)), 4)
		self.assertTupleEqual(tuple(sess.run(tf.shape(images))), (LoadTestCase.N_IMAGES, LoadTestCase.SIZE[1], LoadTestCase.SIZE[0], LoadTestCase.CHANNELS))
		for image in range(LoadTestCase.N_IMAGES):
			for channel in range(LoadTestCase.CHANNELS):
				self.assertAlmostEqual(sess.run(images[image, LoadTestCase.TEST_1_LOCATION[1], LoadTestCase.TEST_1_LOCATION[0], channel]), LoadTestCase.TEST_1_COLOR[channel], delta=LoadTestCase.COLOR_DELTA)
				self.assertAlmostEqual(sess.run(images[image, LoadTestCase.TEST_2_LOCATION[1], LoadTestCase.TEST_2_LOCATION[0], channel]), LoadTestCase.TEST_2_COLOR[channel], delta=LoadTestCase.COLOR_DELTA)

	NAME = "testname"

	def test_load_names(self):
		"""Does load() name the nodes it creates properly?"""
		images = load.load(LoadTestCase.PATH, LoadTestCase.CHANNELS, LoadTestCase.NAME)
		self.assertTrue(images.name.startswith(LoadTestCase.NAME))

if __name__ == "__main__":
	unittest.main()