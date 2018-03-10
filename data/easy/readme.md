# `easy` Dataset

## Summary

This dataset is composed of some plate images and their corresponding CFU counts.

## Organization

### `data` directory

For each plate, there is a correspondingly-named directory. Within a plate's directory, there are 6 single-channel images (PNG, 8-bit, sGray). The channels have the following filenames.

* `0.png`: red channel of original image
* `1.png`: green channel of original image
* `2.png`: blue channel of original image
* `3.png`: inside mask
* `4.png`: edge mask
* `5.png`: outside mask

There is additionally a CFU count (CSV, 1 row, 1 column), which has filename `6.csv`.

### `psd` directory

This directory contains the Photoshop files described in the Methodology section.

### `raw` directory

This directory contains the original images and Excel spreadsheet described in the Methodolgy section.

## Methodology

The plate images were captured with a smartphone (RGB, `2448 × 3264`). The counts were done according to the standard practices of the group. The counts, as well as notes on decisions that were made during counting, were recorded in an Excel spreadsheet.

The plate images were cropped to just the square region containing the plate (`2448 × 2448`). Inside, edge, and outside masks of the images were handdrawn using Photoshop and a graphic tablet ($30, Huion H420). A class mask of an image is a grayscale image of the same dimensions as the original image in which pixels are made black to indicate that the corresponding pixel in the original image is in the class. Pixels are left white if they are not in the class. First, the edge mask was made by marking the edge pixels of each colony (black, Soft Round brush, 3px, 100% hardness, Pin light mode, 100% flow, 10% smoothing) on a new layer. Then, the inside mask was made by filling in each colony with the paint bucket tool (black, Normal mode, 100 tolerance for most colonies, 70 tolerance for some of the colonies on the plate's rim, anti-alias, contiguous, all layers), again on a new layer. Finally, the outside mask was made with the paint bucket tool (black, Normal mode, 100 tolerance, anti-alias, contiguous, all layers) once, again on a new layer. A white background layer was added. Each mask layer, superimposed on the white background and with no other layers visible, was copied (copy merged) and pasted in a new document. This document was converted to the grayscale (sGray color profile). It was then exported (PNG, 8-bit, embedded color profile).

## Issues

* Not all the original images have been masked.
* Some pixels may be shades of gray.
* Some pixels may be claimed by multiple classes.
