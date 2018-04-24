from distutils.core import setup

VERSION = '0.1.7'

setup(
  name = 'cell_counting',
  packages = ['cell_counting', 'cell_counting.models', 'manuscript', 'data'],
  package_data={
    'data': ['*/*']
  },
  version = VERSION,
  description = 'Tools for counting cell colonies on plates of growth medium.',
  author = 'Shea Conlon',
  author_email = 'sheaconlon@berkeley.edu',
  url = 'https://github.com/sheaconlon/cell_counting',
  download_url = 'https://github.com/sheaconlon/cell_counting/archive/v{0:s}'
                 '-alpha.tar.gz'.format(VERSION),
  keywords = ['cell', 'colony', 'count', 'automated', 'CFU', 'image'
              'machine learning', 'artificial intelligence', 'computer vision',
              'segmentation', 'classification'],
  classifiers = [],
  install_requires=[
    'scipy',
    'numpy',
    'openpyxl',
    'tensorflow',
    'scikit-image',
    'matplotlib',
    'imageio',
    'tqdm',
    'psutil',
    'scikit-learn',
    'imgaug'
  ]
)
