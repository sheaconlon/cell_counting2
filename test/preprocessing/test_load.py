import unittest

import tensorflow as tf

from src.preprocessing import load

import os

class LoadTestCase(unittest.TestCase):
	"""Tests for load.py."""

	def test_load_images_and_masks(self):
		"""Does load_images_and_masks() work?"""
		sess = tf.Session()
		images, masks = load.load_images_and_masks(
			os.path.join(os.path.abspath(os.path.dirname(__file__)), "test_load/test_load_images_and_masks"),
			("1", "10")
		)
		self.assertEqual(sess.run(tf.rank(images)), 4)
		self.assertEqual(sess.run(tf.rank(masks)), 4)
		self.assertTupleEqual(tuple(sess.run(tf.shape(images))), (2, 2448, 2448, 3))
		self.assertTupleEqual(tuple(sess.run(tf.shape(masks))), (2, 2448, 2448, 1))
		self.assertTupleEqual(tuple(sess.run(images[0, 1336, 1356, :])), (136, 131, 102))
		self.assertEqual(sess.run(masks[1, 1116, 1468, 0]), 6)

if __name__ == "__main__":
	unittest.main()