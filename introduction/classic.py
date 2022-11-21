# The network input
inputs = [1, 2, 3, 2.5]

# Single neuron example
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2

# 3 neurons example
weights_list = [
    [0.2, 0.8, -0.5, 1],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]
biases = [2, 3, 0.5]

# A single neuron calculation
def neuron(inputs, weights, bias):
    return sum(list(
        map(lambda x, y: x*y, inputs, weights)
    )) + bias


# Map based calculation
def layer(inputs, weights, biases):
    return list(
        map(neuron, [inputs] * len(weights), weights, biases)
    )

# For loop based calculation
def layer_(inputs, weights, biases):
    layer_outputs = []

    for neuron_weights, neuron_bias in zip(weights, biases):
        neuron_output = 0
        for neuron_input, weight in zip(inputs, neuron_weights):
            neuron_output += neuron_input * weight
        neuron_output += neuron_bias
        layer_outputs.append(neuron_output)

# Running the examples
print("Example 01 single neuron: output is {result: .2f}".format(
    result=neuron(inputs, weights, bias)))
print("Example 02 using mapping: outputs are {result}".format(
    result=layer(inputs, weights_list, biases)))
print("Example 02 using loops: outputs are {result}".format(
    result=layer(inputs, weights_list, biases)))
