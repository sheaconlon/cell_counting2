# Random Walks for Image Segmentation

Author: Leo Grady

Published: NOV. 2006

Available at: http://leogrady.net/wp-content/uploads/2017/01/grady2006random.pdf

## Summary
Show the user an image. You know there are $K$ classes of pixels in this image. The user can click and label points, over and over, to refine the classification. Classification is done by analytically solving the following question: From a given point, which user-labeled point would a random walker most likely reach first? It's hard to the random walker to go over sharp gradients. This approach is great for being fast, easy to edit, able to produce any segmentation eventually, and intuitive for users to improve. The math is pretty far out of my reach and this would have to be used a black box.