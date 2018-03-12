#include "Perceptron.h"
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <cmath>
#include <memory>
#include <vector>
namespace ublas = boost::numeric::ublas;

int main(int argc, char const *argv[]) {
  auto sigmoid = [](double x) { return 1 / (1 + exp(-x)); };
  auto dSigmoid = [&sigmoid](double x) {
    return sigmoid(x) * (1 - sigmoid(x));
  };
  auto MLP = std::make_unique<MultiLayerPerceptron>(2, 4, 2, sigmoid, dSigmoid);
  ublas::matrix<double> intput(2, 1, 1);
  auto out = MLP->output(intput); // matrix
  std::cout << "OUT:\n" << out << '\n';
  MLP->train(intput, Matrix(2, 1, 1));
  return 0;
}
