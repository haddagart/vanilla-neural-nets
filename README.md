# Vanilla Python Neural Network
> An implementation of a neural network using raw Python and linear algebra libraries mostly NumPy.
> This implementation is based on the book of : Neural Networks from Scratch by Harrison Kinsley and Daniel Kukiela. [YouTube playlist](https://www.youtube.com/playlist?list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3) is available.

I believe that it is so important to understand what's going under the hood inside the neural networks before attacking any neural network architecture especially that of deep learning.

# Basic notes about neural networks
Check the simple vanilla python implementation using both _map_ and _for-loops_ of a single neuron and a layer of neurons. (The code and examples are based on the reference book). With some originality on using the _map_ function instead of looping through the inputs and weights. [code here](/introduction/classic.py)
## The dot product
### The mathematical background
Given an array of inputs, weights, and biases we can get the outputs as the dot product of weights and inputs added to them the biases. As the formula below shows:
$$y_i = \sum_{i}^{N} w_i * x_i + b_i$$
We say that we use the dot product because the inputs are row-wise and the weights and column wise.

The dot product is doing a sum of multiplications of row _i_ by column _j_ which produces a single value called scalar at position _(i, j)_. That is why the shapes must be convinient for the dot product to be performed correctly. The number of elements in the row must match the number of elements in the column.

You need to understand the shape of inputs and weights and how you are representing the matrices to understand whether you will need to transpose (convert rows to columns) any of your matrices.
### Code implementation in Python + NumPy
```python
    import numpy as np

    # given inputs and weights
    output = np.dot(weights, inputs) + biases
```
> **IMPORTANT**: The order of the dot product is important
## Batches
The network doesn't take the whole inputs set called _dataset_ at once actually the best way to go is to feed the inputs in _batches_ this which is a subset of input examples from the whole _dataset_.

By using batching we will make the learning much easier for the network, and the issue of overfitting may be avoided as it could happen when feeding the network all the examples at once.