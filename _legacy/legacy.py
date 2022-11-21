import numpy as np

np.random.seed(0)

X = [
    [1., 2., 3., 2.5],
    [2., 5., -1., 2.],
    [-1.5, 2.7, 3.3, -0.8]
]

# From NNFS data package
def spiral_data(samples, classes):
    X = np.zeros((samples*classes, 2))
    y = np.zeros(samples*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(samples*class_number, samples*(class_number+1))
        r = np.linspace(0.0, 1, samples)
        t = np.linspace(class_number*4, (class_number+1)*4,
                        samples) + np.random.randn(samples)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y


class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        # Randomly initialize the weights of the layer
        # In this case it is a Gaussian distribution
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.01
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class ReLUActivation:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

X, y = spiral_data(samples=100, classes=3)
dense1 = DenseLayer(2, 3)
activation1 = ReLUActivation()


dense1.forward(X)
activation1.forward(dense1.output)
print(activation1.output[:4])