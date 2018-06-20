import numpy


class Lin_layer():
    def __init__(self, input_number, output_number, lr=0.1):
        self.weigths = numpy.random.rand(output_number, input_number) * 2 - 1
        self.bias = numpy.random.rand() * 2 - 1
        self.lr = lr

    def query(self, input):
        return (numpy.dot(self.weigths, input) + self.bias)[0]

    def train(self, input, layer_out, output_errors):

        self.weigths += (self.lr * numpy.dot(output_errors,
                                             numpy.transpose(input)))
        self.bias += self.lr * output_errors
