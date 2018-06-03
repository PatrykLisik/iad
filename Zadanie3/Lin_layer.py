import numpy


class Lin_layer():
    def __init__(self, input_number, output_number):
        self.weigths = numpy.random.rand(output_number, input_number)
        self.bias = numpy.random.rand()

    def query(self, input):
        return list((numpy.dot(self.weigths, input) + self.bias)[0])

    def train(input, target):
        pass
