# Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift

**Authors:** Sergey Ioffe, Christian Szegedy

**Published:** March 2, 2015

**Available at:** https://arxiv.org/abs/1502.03167

## Summary
This paper solves a problem with neural nets the authors call _internal covariate shift_ using _batch normalization_.

To see what this is, consider a batch of training data in the context of training a neural net. Each batch has different statistics – a different distribution of values. This presents two problems. First, if your learning rate is high, then the neural net will overfit to this batch's distribution. Thus, your learning rate is confined to be low and you are susceptible to overfitting. Second, your activation functions may be prone to saturation, this leading to a vanishing gradient and, again, slow training.

To solve this, the authors recommend using _batch normalization_ between each layer. This must be done with care – as they have described – in order to preserve the nonlinear representation power of the neural net. The authors applied this method to the ImageNet classifcation problem, yielding a 14x reduction in training iterations.

This method is implemented by `tf.layers.batch_normalization`.
