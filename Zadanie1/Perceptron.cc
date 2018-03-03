#include "Perceptron.h"
#include <random>

namespace ublas = boost::numeric::ublas;

// std::srand(std::time(nullptr));

MultiLayerPerceptron::MultiLayerPerceptron(size_t intputNodes,
                                           size_t hiddenNodes,
                                           size_t outputNodes, int BiasH,
                                           int BiasO,
                                           std::function<double(double)> aF)
    : weigthsIntputHidden(ublas::matrix<double>(hiddenNodes, intputNodes)),
      weigthsHiddenOutput(ublas::matrix<double>(outputNodes, hiddenNodes)),
      biasHidden(ublas::matrix<double>(hiddenNodes, 1, BiasH)),
      biasOutput(ublas::matrix<double>(outputNodes, 1, BiasO)),
      ativateFunction(aF) {

  auto randomMatrix = [](ublas::matrix<double> &m, int min, int max) {
    std::uniform_real_distribution<double> unif(min, max);
    std::default_random_engine re;
    for (size_t i = 0; i < m.size1(); i++)
      for (size_t j = 0; j < m.size2(); j++)
        m(i, j) = unif(re);
  };

  randomMatrix(weigthsHiddenOutput, -1, 1);
  randomMatrix(weigthsIntputHidden, -1, -1);
}
