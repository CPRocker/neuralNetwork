import numpy as np
from cd import createData

class LayerDense:
    def __init__(self, nInputs, nNeurons):
        self.weights = 0.1 * np.random.randn(nInputs, nNeurons)
        self.biases = np.zeros((1, nNeurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class ActivationReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class ActivationSoftmax:
    def forward(self, inputs):
        expValues = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))
        probabilities = expValues / np.sum(expValues, axis = 1, keepdims = True)
        self.output = probabilities

X, y = createData(100, 3)

dense1 = LayerDense(2, 3)
activation1 = ActivationReLU()

dense2 = LayerDense(3, 3)
activation2 = ActivationSoftmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])
