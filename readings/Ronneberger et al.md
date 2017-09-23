# U-Net: Convolutional Networks for Biomedical Image Segmentation

Authors: Olaf Ronneberger, Philipp Fischer, Thomas Brox

Submitted: 18 May 2015

Available at: https://arxiv.org/abs/1505.04597

## Summary
This paper was applied to a sort of different domain, biomedical images, but a pretty similar task. They used a U-shaped architecture (convolution/transfer/downsampling then convolution/transfer/upsampling) with skipping/copying of outputs from the downsample side to the upsample side.

## Ideas
1. **high loss function weight on separators**: can weight separator errors highly in order to focus on segmentation of touching units, not too much just in/out classification
2. **weight initialization procedure**: $sqrt(2/N)$, where $N$ is the number of incoming nodes of a neuron, to produce unit variance in each feature map
3. **augmentation procedure**: shift, rotation, deformation (esp elastic random), gray value variation
4. **skipping**: allows for better use of location and context in upsampling (not just semantics)
5. **upsampling**: allows for "probability heatmap" instead of just classification
