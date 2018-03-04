#ifndef MLP_H
#define MLP_H
#include <boost/numeric/ublas/matrix.hpp>
#include <functional>

namespace ublas = boost::numeric::ublas;
typedef ublas::matrix<double> Matrix;
class MultiLayerPerceptron {
private:
  Matrix weigthsIntputHidden;
  Matrix weigthsHiddenOutput;
  Matrix biasHidden;
  Matrix biasOutput;
  std::function<double(double)> activationFunction; // probably sigmoid function

public:
  MultiLayerPerceptron(size_t intputNodes, size_t hiddenNodes,
                       size_t outputNodes, int BiasH, int BiasO,
                       std::function<double(double)> aF);
  Matrix output(Matrix intput);
  void train(Matrix intput, Matrix output);
  virtual ~MultiLayerPerceptron() = default;
};
#endif /* end of include guard:MultiLayerPerceptron */
