import unittest

import tensorflow as tf

from src.preprocessing import load, patchify

import os

class PatchifyTestCase(unittest.TestCase):
	"""Tests for patchify.py."""

	IMAGES = tf.constant(
		[
			[
				[[1]*3, [2]*3, [3]*3, [4]*3],	
				[[5]*3, [6]*3, [7]*3, [8]*3],
				[[9]*3,[10]*3,[11]*3,[12]*3]
			],
			[
				[[1]*3, [2]*3, [3]*3, [4]*3],	
				[[5]*3, [6]*3, [7]*3, [8]*3],
				[[9]*3,[10]*3,[11]*3,[12]*3]
			]
		]
	)

	MASKS = tf.constant(
		[
			[
				[[1], [2], [3], [4]],	
				[[5], [6], [7], [8]],
				[[9],[10],[11],[12]]
			],
			[
				[[1], [2], [3], [4]],	
				[[5], [6], [7], [8]],
				[[9],[10],[11],[12]]
			]
		]
	)

	def test_patchify(self):
		"""Does patchify() work?"""
		sess = tf.Session()
		patches, classes = patchify.patchify(PatchifyTestCase.IMAGES, PatchifyTestCase.MASKS, 3, 1)
		self.assertEqual(sess.run(classes[2]), 6)
		self.assertTupleEqual(tuple(sess.run(patches[2, 0, 0, :])), (1, 1, 1))

if __name__ == "__main__":
	unittest.main()