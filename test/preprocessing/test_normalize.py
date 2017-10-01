import unittest

import tensorflow as tf

from src.preprocessing import load, normalize

class NormalizeTestCase(unittest.TestCase):
	"""Tests for load.py."""

	DELTA = 10e-4

	TYPICAL_IMAGES = tf.constant([
		[
			[[1, 100], [2, 100],  [3, 100],  [4, 100]  ],
			[[8, 100], [7, 100],  [6, 100],  [5, 100]  ],
			[[9, 100], [10, 100], [11, 100], [100, 100]],
		],
		[
			[[100, 100], [100, 100], [100, 100], [100, 100]],
			[[100, 100], [100, 100], [100, 100], [100, 100]],
			[[100, 100], [100, 100], [100, 100], [100, 100]],
		],
	])
	TYPICAL_WINDOW = 3
	TYPICAL_PADDING = "SYMMETRIC"

	# Median is (6 + 7) / 2 = 6.5.
	# After diving by the median, we should have:
	# [[
	# 	[[1/6.5], [2/6.5],  [3/6.5],  [4/6.5]  ],
	# 	[[8/6.5], [7/6.5],  [6/6.5],  [5/6.5]  ],
	# 	[[9/6.5], [10/6.5], [11/6.5], [100/6.5]],
	# ]]
	# Consider the entry at the far lower-right.
	# Local mean with SYMMETRIC padding will be
	# 	(6/6.5 + 5/6.5 + 5/6.5
	#    + 11/6.5 + 100/6.5 + 100/6.5
	#    + 11/6.5 + 100/6.5 + 100/6.5) / 9 = 7.487.
	TYPICAL_NORMALIZED_LOWER_RIGHT = 100/6.5 - 7.487

	# Consider the entry in the middle row, second column.
	# Local mean will be
	# 	(1/6.5 + 2/6.5 + 3/6.5 + 8/6.5 + 7/6.5 + 6/6.5
	#		+ 9/6.5 + 10/6.5 + 11/6.5) / 9 = 0.974.
	TYPICAL_NORMALIZED_MIDDLE_SECOND = 7/6.5 - 0.974

	def test_smdm_normalize_typical(self):
		"""Does smdm_normalize() produce images with median about 0 and mean about 1?"""
		sess = tf.Session()
		normalized = sess.run(normalize.smdm_normalize(NormalizeTestCase.TYPICAL_IMAGES, NormalizeTestCase.TYPICAL_WINDOW, NormalizeTestCase.TYPICAL_PADDING))
		self.assertAlmostEqual(normalized[0, 1, 1, 0], NormalizeTestCase.TYPICAL_NORMALIZED_MIDDLE_SECOND, delta=NormalizeTestCase.DELTA)
		self.assertAlmostEqual(normalized[0, 2, 3, 0], NormalizeTestCase.TYPICAL_NORMALIZED_LOWER_RIGHT, delta=NormalizeTestCase.DELTA)

	NAME = "testname"

	def test_smdm_normalize_names(self):
		"""Does smdm_normalize() name at least the node it returns properly?"""
		normalized = normalize.smdm_normalize(NormalizeTestCase.TYPICAL_IMAGES, NormalizeTestCase.TYPICAL_WINDOW, NormalizeTestCase.TYPICAL_PADDING, NormalizeTestCase.NAME)
		self.assertTrue(normalized.name.startswith(NormalizeTestCase.NAME))


if __name__ == "__main__":
	unittest.main()