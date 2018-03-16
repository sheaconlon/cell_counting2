# `multicondition` Dataset

## Summary

This dataset is composed of some sets of plate images and their CFU counts.

## Organization

### `data` directory

Each plate has a correspondingly-named subdirectory. Within a plate's subdirectory, there is a set of 3 RGB images (PNG, 2448px x 2448px), described below.

* `light_uncovered_far_noperspective.png`: an image of the plate taken with extra lighting, with the plate's cover removed, from a rather far distance, with no perspective
* `nolight_uncovered_close_minorperspective.png`: an image of the plate taken without extra lighting, with the plate's cover removed, from a close distance, with minor perspective
* `light_covered_close_severeperspective.png`: an image of the plate taken with extra lighting, with the plate's cover left on, from a close distance, with severe perspective

There is additionally a CFU count (CSV, 1 row, 1 column), which has filename `count.csv`.

### `raw` directory

This directory contains the original cropped images and counts.

## Methodology

The plate images were captured with a smartphone (RGB, `3036px x 4048px`). The counts were made by Shea (so not necessarily according to the standard practices of the group).

The plate images were cropped to just the square region containing the plate. The given `resize.py` script was used to resize them to `2448px x 2448px`.
