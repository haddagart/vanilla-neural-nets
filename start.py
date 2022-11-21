from neuralnets.layers import Dense
from neuralnets.activations import ReLU, SoftMax
from neuralnets.losses import CategoricalCrossEntropy
from datasets.scatters import spiral_data
import numpy as np

np.random.seed(0)

# Define the model
def nn():
    return [
        Dense(2, 3),
        ReLU(),
        Dense(3, 3),
        SoftMax(),
    ]
# Generate the dataset
X, y = spiral_data(samples=100, classes=3)

# Call the model and do the training
model = nn()
model[0].forward(X)
for i in range(1, len(model)):
    model[i].forward(model[i-1].output)

# Calculate the loss
loss_function = CategoricalCrossEntropy()
loss_value = loss_function.calculate(model[-1].output, y)

print(loss_value)