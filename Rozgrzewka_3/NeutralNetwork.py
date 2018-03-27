import numpy
import scipy.special


class NeutralNetwork:

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learningrate):
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        self.lr = learningrate

        self.wih = numpy.random.rand(self.hnodes, self.inodes)
        self.bih = numpy.random.rand(self.hnodes, 1)
        self.who = numpy.random.rand(self.onodes, self.hnodes)
        self.bho = numpy.random.rand(self.onodes, 1)
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    def train(self, input_list, target_list):
        inputs=numpy.array(input_list, ndmin=2).T
        targets=numpy.array(target_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_inputs += self.bih
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_inputs += self.bho
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)

        self.who += self.lr * numpy.dot((output_errors * final_outputs *
                                        (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs))
        self.bho += self.lr * output_errors * final_outputs * (1.0 -
                                                               final_outputs)
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs *
                                        (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs))
        self.bih += self.lr * hidden_errors * hidden_outputs * (1.0 -
                                                                hidden_outputs)

    def query(self, input_list):
        inputs=numpy.array(input_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_inputs += self.bih
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_inputs += self.bho
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
