#include "Perceptron.h"
#include <algorithm>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <iostream>
#include <random>
#define DEBUG 1

namespace ublas = boost::numeric::ublas;

// std::srand(std::time(nullptr));

MultiLayerPerceptron::MultiLayerPerceptron(size_t intputNodes,
                                           size_t hiddenNodes,
                                           size_t outputNodes, int BiasH,
                                           int BiasO,
                                           std::function<double(double)> aF)
    : weigthsIntputHidden(Matrix(hiddenNodes, intputNodes)),
      weigthsHiddenOutput(Matrix(outputNodes, hiddenNodes)),
      biasHidden(Matrix(hiddenNodes, 1, BiasH)),
      biasOutput(Matrix(outputNodes, 1, BiasO)), activationFunction(aF) {

  auto randomMatrix = [](Matrix &m, double min, double max) {
    std::random_device
        rd; // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<double> dis(min, max);
    std::transform(m.data().begin(), m.data().end(), m.data().begin(),
                   [&](double) { return dis(gen); });

  };
  randomMatrix(weigthsHiddenOutput, -1, 1);
  randomMatrix(weigthsIntputHidden, -1, 1);

  if (DEBUG) {
    std::cout << "weigthsHiddenOutput" << weigthsHiddenOutput
              << "\nweigthsIntputHidden" << weigthsIntputHidden << '\n';
  }
}

// assume that given vector is INTPUT_NUMBER X N
Matrix MultiLayerPerceptron::output(Matrix intput) {
  // hidden output
  Matrix hidden = ublas::prod(weigthsIntputHidden, intput);

  if (DEBUG) {
    std::cout << "HIDDEN=weigthsIntputHidden X intput" << '\n'
              << hidden << "\n=\n"
              << weigthsIntputHidden << "\nX\n"
              << intput << "\nHIDDEN+=biasHidden" << hidden << "\n+\n"
              << biasHidden << '\n';
  }

  hidden += biasHidden;

  if (DEBUG) {
    std::cout << "=\n" << hidden << '\n';
  }

  // hidden with activationFunction
  Matrix hidden_aF = hidden;
  std::transform(hidden_aF.data().begin(), hidden_aF.data().end(),
                 hidden_aF.data().begin(), activationFunction);
  // bad works on copy
  // std::for_each(hidden.data().begin(), hidden.data().end(),
  // activationFunction);

  if (DEBUG) {
    std::cout << "HIDDEN with activationFunction applied to\n"
              << hidden_aF << '\n';
  }

  // output
  Matrix output = ublas::prod(weigthsHiddenOutput, hidden_aF);
  output += biasOutput;
  std::transform(output.data().begin(), output.data().end(),
                 output.data().begin(), activationFunction);

  return output;
}

void MultiLayerPerceptron::train(Matrix in, Matrix ans) {

  Matrix hidden = ublas::prod(weigthsIntputHidden, in);
  hidden += biasHidden;
  // Conpute error matrixes
  // ERROR=ans-output
  Matrix out = output(in);
  Matrix errorOutput = ans - out;

  // gradient Hidden->output
  // hiden output is what hidden layer giver to output layer
  // hidden_output=weigthsIntputHidden X intput + biasHidden
  // gradinet_HO=lr*E_mx X dAF(output_output) X transpose(hidden_output)
}
