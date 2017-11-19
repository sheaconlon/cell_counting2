from scipy import ndimage
import numpy as np
import tempfile
import openpyxl
import os
from scipy import misc
import random

class Dataset(object):
	"""A dataset consisting of some examples of input/output pairs. Minimizes
	memory usage by keeping most of the dataset on disk at any given time.

	Note that if there are extra examples beyond a multiple of SEGMENT_SIZE,
		then these examples will be ignored.
	"""

	SEGMENT_SIZE = 5

	def __init__(self):
		"""Create a dataset."""
		self._segments = 0
		self._segment_dir = tempfile.TemporaryDirectory()

	class ImageIntegerIterator(object):
		def __init__(self, image_dir_path, labels, shape):
			self._image_dir_path = image_dir_path
			self._labels = labels
			self._shape = shape
			self._image_name_iterator = iter(os.listdir(self._image_dir_path))

		def __iter__(self):
			return self

		def __next__(self):
			image_name = next(self._image_name_iterator)
			image_path = os.path.join(self._image_dir_path, image_name)
			image = ndimage.imread(image_path)
			image = misc.imresize(image, self._shape, interp="bilinear")
			image_name_noext = ".".join(image_name.split(".")[:-1])
			label = self._labels[image_name_noext]
			return (image, label)

	def load_images_and_excel_labels(self, image_dir_path, label_sheet_path,
			filename_col, label_col, shape):
		"""Makes the inputs be the pixels of the images in some directory and
			the labels be given by an Excel spreadsheet mapping those images'
			filenames to integers.

		Args:
			image_dir_path (str): The path to the directory containing the
				images.
			label_sheet_path (str): The path to the spreadsheet containing the
				labels.
			filename_col (str): The column in the spreadsheet that contains
				the filenames.
			label_col (str): The column in the spreadsheet that contains the
				labels.
			shape (tuple of int): The shape to resize the images to. Should be
				a 3-element tuple of height, width, and channel depth.

		"""
		labels = self._load_excel_mapping(label_sheet_path, filename_col,
			label_col)
		example_iterator = self.ImageIntegerIterator(image_dir_path, labels,
			shape)
		self._load_examples(example_iterator)

	def get_batch(self, size):
		"""Get a batch of examples.

		Args:
			size (int): The number of examples.

		Returns:
			(tuple of np.ndarray): The inputs and outputs of the batch. Each
				array has size rows, where the i-th row corresponds to the i-th
				example.
		"""
		segments_needed = int(size / self.SEGMENT_SIZE) + 1
		chosen_segments = random.sample(range(self._segments), segments_needed)
		cache = []
		for segment in chosen_segments:
			this_segment_dir = os.path.join(self._segment_dir.name,
				str(segment))
			inputs = np.load(os.path.join(this_segment_dir, "inputs.npy"))
			outputs = np.load(os.path.join(this_segment_dir, "outputs.npy"))
			cache.append((inputs, outputs))
		chosen_examples = random.sample(
			range(segments_needed * self.SEGMENT_SIZE), size)
		inputs = []
		outputs = []
		for example in chosen_examples:
			segment = int(example / self.SEGMENT_SIZE)
			row = example % self.SEGMENT_SIZE
			inputs.append(cache[segment][0][row, ...])
			outputs.append(cache[segment][1][row, ...])
		inputs = np.stack(inputs, axis=0)
		outputs = np.stack(outputs, axis=0)
		return inputs, outputs

	def close(self):
		"""Close this dataset. This dataset will not be useable afterward.

		This method must be called because it deletes temporary files on disk.
		"""
		self._segment_dir.__exit__()
		self._segment_dir = None

	def _load_examples(self, example_iterator):
		inputs, outputs = [], []
		for i, example in enumerate(example_iterator):
			input, output = example
			inputs.append(input)
			outputs.append(output)
			if (i + 1) % self.SEGMENT_SIZE == 0:
				self._save_segment(inputs, outputs)
				inputs, outputs = [], []

	def _save_segment(self, inputs, outputs):
		inputs = np.stack(inputs, axis=0)
		outputs = np.stack(outputs, axis=0)
		this_segment_dir = os.path.join(self._segment_dir.name,
			str(self._segments))
		os.mkdir(this_segment_dir)
		np.save(os.path.join(this_segment_dir, "inputs.npy"), inputs)
		np.save(os.path.join(this_segment_dir, "outputs.npy"), outputs)
		self._segments += 1

	@staticmethod
	def _load_excel_mapping(path, key_col, value_col):
		mapping = {}
		wb = openpyxl.load_workbook(path)
		ws = wb.active
		i = 2
		while True:
			key = ws[key_col+str(i)].value
			val = ws[value_col+str(i)].value
			if key is None or val is None:
				return mapping
			mapping[key] = val
			i += 1