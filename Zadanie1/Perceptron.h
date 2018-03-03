#ifndef MLP
#define MLP
#include <boost/numeric/ublas/matrix.hpp>
#include <functional>
namespace ublas = boost::numeric::ublas;
class MultiLayerPerceptron {
private:
  ublas::matrix<double> weigthsIntputHidden;
  ublas::matrix<double> weigthsHiddenOutput;
  ublas::matrix<double> biasHidden;
  ublas::matrix<double> biasOutput;
  std::function<double(double)> ativateFunction; // probably sigmoid function

public:
  MultiLayerPerceptron(size_t intputNodes, size_t hiddenNodes,
                       size_t outputNodes, int BiasH, int BiasO,
                       std::function<double(double)> aF);
  virtual ~MultiLayerPerceptron() = default;
};
#endif /* end of include guard:MultiLayerPerceptron */
