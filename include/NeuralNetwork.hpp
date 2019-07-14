#ifndef neuralnetwork_hpp
#define neuralnetwork_hpp

#include <random>
#include <vector>

#include "Matrix.hpp"

class NeuralNetwork
{
public:
    NeuralNetwork();
    NeuralNetwork(NeuralNetwork const& nn);
    NeuralNetwork(std::vector<int> layers);
    ~NeuralNetwork();

    Matrix predict(Matrix const& m) const;
    void mutate(double factor);

    std::pair<std::vector<Matrix>,std::vector<Matrix>> dump() const;
    void set(std::pair<std::vector<Matrix>,std::vector<Matrix>>);

private:
    Matrix matmul(Matrix const& x, Matrix const& y) const;
    Matrix activation(Matrix const& m) const;
    std::vector<Matrix> weight;
    std::vector<Matrix> bias;
    bool initialized = false;
};

#endif