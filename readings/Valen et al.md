# Deep Learning Automates the Quantitative Analysis of Individual Cells in Live-Cell Imaging Experiments

Authors: David A. Van Valen, Takamasa Kudo, Keara M. Lane, Derek N. Macklin, Nicolas T. Quach, Mialy M. DeFelice, Inbal Maayan, Yu Tanouchi, Euan A. Ashley, Markus W. Covert

Published: November 4, 2016

Available at: https://doi.org/10.1371/journal.pcbi.1005177

## Summary
This paper uses conv-nets to classify pixels as "background", "cell border", or "cell interior". The result is a transformed image which can, for example, be fed to image segmentation algorithms and used to get a cell colony count. They endeavored to decrease training data curation time,  improve segmentation accuracy, and make solution sharing easier. They considered two problems: mammalian cytoplasm segmentation (least well solved JI ~ 0.8), bacterial cytoplasm segmentation (pretty well solved JI ~ 0.95), and mammalian nucleus segmentation (well solved JI ~ 0.9). A conv-net basically consists of some rounds of convolution/transfer/pooling, then a fully-connected stage.

## Ideas
1. **cell heterogeneity**: extend to cell type classification by creating different classes for each cell type – then you get to extract statistics on the various populations and how their characteristics differ and vary over time
2. **evolution over time**: extend to frame sequences to analyze patterns over time – segmentation and (linear assignment problem) suffice, really helps in accuracy assessment, gives you single-cell growth rate data
3. **normalization**: images should be normalized  to yield insensitivity to class brightness (so it's transferrable to differing lighting), here subtract median and divide mean
4. **post-processing refinement**: after classification, active contours or another method can be used to finalize the segmentation (suggested looking into conditional random fields)
5. **receptive field size**: highly sensitive to patch size, should be on order of cell diameter, but again it's highly sensitive
6. **d-regularly sparse kernels**: method to do convolution on each pixel of an image without doing a ton of repeated work, very important to make it feasible
7. **lab(maybe X cell)-specific models**: authors believe the most potential lies in lab(maybe X cell)-specific models, rather than all-purpose models
8. **Jaccard index benchmarking**: can benchmark image segmentations using reference segmentations and a formula
9. **confidence scores**: can give confidence score (class i) / (sum of classes i), errors will have low confidence scores!
10. **not much data needed**: not much original data needed if done right! 100 mammalian cells, 500 nuclei, or 300 bacteria though lots of labor
11. **multi-resolution**: feed multiple spatial scales as features

## Degrees of Freedom
1. patch size (per task), 31 bacteria and 61 mammalian cytoplasm/nuclei here
2. number of rounds, 4-6 here
3. number of filters (per round?), ?? here
4. size of filters (per round?), 3-5 here
5. stride of filters (per round?), ?? here
6. transfer function (per round?), - ReLU here
7. down-sampling size (per round?), - 2 x 2 here
8. down-sampling stride (per round?), - 2 here
9. number of fully-connected layers, ?? here
10. size of fully-connected layers (per layer), ?? here
11. activation function of fully-connected layers (per layer?), ?? here
12. cost function, softmax here
13. regularization, L2 and dropout and batch normalization here
14. weight initialization technique, random (otherwise unspecified) here
15. descent technique, RMSprop or BGD here
16. post-processing step (per task), thresholding for bacterial cytoplasm/mammalian nucleus and nuclear prediction+active contours for mammalian cytoplasm here
17. generic or cell-specific models, cell-specific here
18. normalization technique, here subtract median and divide mean
19. data augmentation methods, rotation and reflection but not shearing here
20. features, multi-resolution base pixels here+nuclei labeled for mammalian cytoplasm
21. training dataset class distribution, artificially equalized here
22. ensembling techniques, mean of 5 here
