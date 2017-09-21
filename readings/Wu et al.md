# Early Hierarchical Contexts Learned by Convolutional Networks for Image Segmentation

Authors: Zifeng Wu, Yongzhen Huang, Yinan Yu, Liang Wang, and Tieniu Tan

Published: 08 December 2014

Available at: http://www.cbsr.ia.ac.cn/users/ynyu/icpr2014.pdf

## Summary
This paper seeks to segment people from an image. Their main point is that they used multiple size patches, convolved each, and fed all of this into a multiplayer perceptron.

## Ideas
1. **multiple size patches**: better consideration of context if you use multiple patch sizes and let these be learned jointly – maybe not as important for cells on a plate as for people outdoors, though