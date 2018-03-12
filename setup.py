from distutils.core import setup

setup(
  name = 'cell_counting',
  packages = ['cell_counting'],
  version = '0.1.1',
  description = 'Tools for counting cell colonies on plates of growth medium.',
  author = 'Shea Conlon',
  author_email = 'sheaconlon@berkeley.edu',
  url = 'https://github.com/sheaconlon/cell_counting',
  download_url = 'https://github.com/sheaconlon/cell_counting/archive/v0.09-alpha.tar.gz',
  keywords = ['cell', 'colony', 'counting', 'automated', 'machine learning'],
  classifiers = [],
  install_requires=[
    'scipy',
    'numpy',
    'openpyxl',
    'tensorflow',
    'scikit-image',
    'matplotlib'
  ]
)
