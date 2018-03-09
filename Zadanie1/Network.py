import numpy
import scipy.special


class NeuralNetwork:

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learningrate):
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        self.lr = learningrate

        self.wih = numpy.random.rand(self.hnodes, self.inodes) * 2 - 1
        self.who = numpy.random.rand(self.onodes, self.hnodes) * 2 - 1
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    def train(self, inputs, targets):
        hidden_inumpyuts = numpy.dot(self.wih, inumpyuts)
        hidden_outputs = self.activation_function(hidden_inumpyuts)
        final_inumpyuts = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inumpyuts)
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)
        self.who += self.lr * numpy.dot((output_errors * final_outputs *
                                        (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs *
                                        (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs))
        pass

    def query(self, inumpyuts):
        hidden_inumpyuts = numpy.dot(self.wih, inumpyuts)
        hidden_outputs = self.activation_function(hidden_inumpyuts)
        final_inumpyuts = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inumpyuts)
        return final_outputs


nn = NeuralNetwork(3, 3, 3, 0.3)
inumpyuts = numpy.array([1.0, 0.5, -1.5])
outputs = nn.query(inumpyuts)
print(outputs)
