# What's a vision transformer and why are they outperforming CNNs?

1. TOC
{:toc}

## Convolutional neural networks

As part of doing the fast.ai course, I've learnt about convolutional neural networks. Up until very recently, this was the state of the art architecture for solving image classification problems.

They work by utilising layers that instead of multiplying the entire image by a weight for each pixel, instead convolve a kernel (which is just a matrix of values) over the whole image. This process is explained visually below, in a snippet from the [fast.ai](https://nbviewer.org/github/fastai/fastbook/blob/master/13_convolutions.ipynb) book. The kernel is evaluated at any point on the image by multiplying the relevant value in the kernel by the associated pixel and summing the results.

![](/images/conv.png "Example of how convolution works, from the fast.ai course, chapter 13")

This allows the models to detect features in an image, such as edges. The following kernel allows for the detection of top edges. That's because if there's low values at the top, and high values at the bottom, the computed value will be high.

```python
tensor([-1, -1, -1],
       [ 0,  0,  0],
       [ 1,  1,  1])
```

Neural networks such as [ResNet](https://arxiv.org/abs/1512.03385) are able to achieve very high accuracy when trained on the ImageNet test set, with only a 3.57% error rate.

## Vision transformers

A vision transformer is a new snazzy type of architecture that's proving to perform better than convolutional networks. It applies a transformer architecture, commonly used for natural language processing models, to image recognition.

### What's a transformer?

Prior to doing this course, I knew slightly about convolutional nets, but I'd never heard of a transformer before. The only thing I know is that ChatGPT uses them, hence the T (Generative Pre-trained Transformer).

According to [this paper](https://arxiv.org/pdf/1706.03762.pdf), the inventors of the transformer, they look like this:

![](/images/transformer.png "The transformer architecture")

So what does that do? It's broken into two distinct parts, the encoder (left side) and decoder (right side). In short, the encoder maps an input sequence of symbol representations (such as word tokens) to a sequence of continuous representations. The decoder generates an output sequence of symbols.

The transformer adds an extra detail to this encoder-decoder model, called attention. This is a mechanism for indicating other important parts of a sequence that are important to each step in the sequence. It's a way of encoding semantics through context, for example thinking about other words when reading a particular word in a sentence.

### How does a vision transformer work?

So transformers are good for language tasks such as translation because they convert an input sequence into an output sequence. How does this work with images?

The trick is to treat an image like a sentence. That is, split it up into patches and assign each patch a position.

For example, a sequence of sentence tokens would look like this:

```
 1    2     3    4    5    6    7   8    9
The quick brown fox jumps over the lazy dog.
```

And an image can be deconstructed in the same way, as seen in the [paper on vision transformers](https://arxiv.org/abs/1512.03385):

![](/images/vit.png "Splitting up of images into patches and the vision transformer architecture")

So the sequence of patches is flattened, assigned a position, and passed to the transformer. A classification token is added to the sequence to allow for learning of the class of the image. A MLP head is attached to the output to perform the final classification step.

## Comparison

The [model from the paper](https://arxiv.org/abs/1512.03385) showed state of the art performance by beating all of the existing architecture on a series of image data sets.

While not entirely explainable, the paper suggests that its architecture removes image-specific inductive biases - meaning for example that pixels close together are related to each other. Because of the attention layers, the vision transformer can relate pixels to the entirety of the image, whereas a convlution kernel will only work on pixels in the same area.

It was also shown in the paper that the vision transformer is much cheaper to train than a convolutional network - a very important factor to consider when deciding on model architecture.

It will be interesting to see how this type of model develops into the future. The paper's authors also mention some problems to be addressed such as figuring out how to apply the vision transformer to segmentation and detection problems.