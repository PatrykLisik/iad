#include "Perceptron.h"
#include <algorithm>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <iostream>
#include <random>
#define DEBUG 0

namespace ublas = boost::numeric::ublas;

MultiLayerPerceptron::MultiLayerPerceptron(size_t intputNodes,
                                           size_t hiddenNodes,
                                           size_t outputNodes,
                                           std::function<double(double)> aF,
                                           std::function<double(double)> daF)
    : weigthsIntputHidden(Matrix(hiddenNodes, intputNodes)),
      weigthsHiddenOutput(Matrix(outputNodes, hiddenNodes)),
      biasHidden(Matrix(hiddenNodes, 1)), biasOutput(Matrix(outputNodes, 1)),
      activationFunction(aF), dActivationFunction(daF) {

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
  randomMatrix(biasHidden, -1, 1);
  randomMatrix(biasOutput, -1, 1);

  if (DEBUG) {
    std::cout << "weigthsHiddenOutput" << weigthsHiddenOutput
              << "\nweigthsIntputHidden" << weigthsIntputHidden << '\n';
  }
}

// assume that given vector is INTPUT_NUMBER X N
Matrix MultiLayerPerceptron::output(Matrix intput) {
  // calculate signals into hidden layer
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

  // calculate the signals emerging from hidden layer
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

  // compute hidden layer
  Matrix hidden = ublas::prod(weigthsIntputHidden, in);
  hidden += biasHidden;
  Matrix hidden_aF = hidden;
  std::transform(hidden_aF.data().begin(), hidden_aF.data().end(),
                 hidden_aF.data().begin(), activationFunction);
  // compute output layer
  // output without activationFunction applied is needed later
  Matrix output = ublas::prod(weigthsHiddenOutput, hidden_aF);
  output += biasOutput;
  Matrix outputAf = output;
  std::transform(output.data().begin(), output.data().end(),
                 outputAf.data().begin(), activationFunction);

  // Conpute error matrixes

  // ERROR_output=ans-output
  // outputAf final output
  Matrix errorOutput = ans - outputAf;
  Matrix outputdAf = output;

  // applay derivative of activationFunction
  std::transform(output.data().begin(), output.data().end(),
                 outputdAf.data().begin(), dActivationFunction);
  // compute output->hidden adjustment
  std::cout << "HERE" << errorOutput << '\n' << outputdAf << '\n';
  Matrix tmp = ublas::prod(errorOutput, outputdAf);
  Matrix deltaHO = lr * ublas::prod(tmp, ublas::trans(hidden_aF));
  // learn!
  weigthsHiddenOutput += deltaHO;

  // ERROR_hidden=(weigthsHiddenOutput)^1 X ERROR_output
  Matrix hiddenDaF = hidden;
  std::transform(hidden.data().begin(), hidden.data().end(),
                 hiddenDaF.data().begin(), dActivationFunction);
  Matrix errorHidden =
      ublas::prod(ublas::trans(weigthsHiddenOutput), errorOutput);
  // deltaHI=le * ERROR_hidden X DaF(hidden) X trans(inputs)
  tmp = ublas::prod(errorHidden, hiddenDaF);
  weigthsIntputHidden += lr * ublas::prod(tmp, ublas::trans(in));
}
