import numpy as np


class ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class SoftMax:
    def forward(self, inputs):
        # The subtraction is designed to avoid getting large values in the exp(x)
        # Thus the calculation is going to be sure and faster

        exp_vals = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probs = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        self.output = probs
