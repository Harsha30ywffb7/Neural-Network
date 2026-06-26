import numpy as np
class Nueron:
    def __init__(self, n_inputs):
        #initialise weights and bias
        self.weights = np.random.randn(n_inputs)
        self.bias = np.random.randn()

    def forward(self, x):
        #linear combination
        z = np.dot(self.weights, x) + self.bias

        # applying ReLU
        return self.ReLU(z)

    def ReLU(self, z):
        return np.max(0, z)