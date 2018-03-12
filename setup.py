# from setuptools import setup, find_packages
#
# setup(
#     name='cell_counting',
#     version='0.1',
#     description='A sample Python project',
#     packages=find_packages()
# )
from distutils.core import setup

setup(
  name = 'cell_counting',
  packages = ['cell_counting'],
  version = '0.1',
  description = 'Tools for counting cell colonies on plates of growth medium.',
  author = 'Shea Conlon',
  author_email = 'sheaconlon@berkeley.edu',
  url = 'https://github.com/sheaconlon/cell_counting',
  download_url = 'https://github.com/sheaconlon/cell_counting/archive/v0.09-alpha.tar.gz',
  keywords = ['cell', 'colony', 'counting', 'automated', 'machine learning'],
  classifiers = [],
)
