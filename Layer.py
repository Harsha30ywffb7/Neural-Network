from Neuron import Neuron
import numpy as np
class Layer:
    def __init__(self, inputs, n_neurons):
        self.nuerons =[]
        for _ in range(n_neurons):
            self.nuerons.append(Neuron(inputs))

    def forward(self, x):
        outputs =[]
        for neuron in self.nuerons:
            outputs.append(neuron.forward(x))
        return np.array(outputs)


class NueralNetwork:
    def __init__(self):
        self.hidden_layer1 = Layer(inputs=3, n_neurons=3)
        self.hidden_layer2 = Layer(inputs=3, n_neurons=2)

        self.output_layer = Layer(inputs=2, n_neurons=1)


    def forward(self, x):
        out1 = self.hidden_layer1.forward(x)
        print("Hidden layer 1:", out1)

        out2 = self.hidden_layer2.forward(out1)
        print("Hidden layer 2:", out2)
        final_output = self.output_layer.forward(out2)
        print("Final output:", final_output)

        return final_output
           

            

