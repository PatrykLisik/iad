import numpy
import scipy.special


class NeuralNetwork:

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learningrate):
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        self.lr = learningrate

        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5),
                                       (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5),
                                       (self.onodes, self.hnodes))
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    def train(self, inputs, targets):
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)
        self.who += self.lr * numpy.dot((output_errors * final_outputs *
                                        (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs *
                                        (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs))
        pass

    def query(self, inputs):
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs


input_nodes = 1
hidden_nodes = 3
output_nodes = 1
learningrate = 0.1
nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learningrate)

input_list = [0]
target_list = [0]

input_file = open("approximation_train_1.txt", "r")
for line in input_file:
    buff = line.split(" ")
    input_buff = float(buff[0])
    target_buff = float(buff[1])
    input_list[0] = input_buff
    target_list[0] = target_buff
    nn.train(numpy.array(input_list, ndmin=2).T,
             numpy.array(target_list, ndmin=2).T)

input_file.close()

input_file = open("approximation_test.txt", "r")
for line in input_file:
    buff = line.split(" ")
    input_buff = float(buff[0])
    input_list[0] = input_buff
    print(nn.query(numpy.array(input_list, ndmin=2).T))

input_file.close()
