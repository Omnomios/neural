#ifndef neuralnetwork_hpp
#define neuralnetwork_hpp

#include <random>
#include <vector>
#include <valarray>
#include "LossFunction.hpp"

class NeuralNetwork
{
public:
    struct Layer
    {
        std::valarray<double> value;
        std::valarray<double> bias;
        std::valarray<double> delta;
        std::vector<
            std::valarray<double>
        > weight;
        Layer(int neurons, int links) : value(neurons), bias(neurons), delta(neurons), weight(neurons, std::valarray<double>(links)){}
    };

public:
    // Lifecycle
    NeuralNetwork();
    NeuralNetwork(NeuralNetwork const& rhs);
    NeuralNetwork(std::vector<int> layers);
    ~NeuralNetwork();

    // State management
    std::vector<Layer> dump() const;
    void restore(std::vector<Layer>);

    // Network functions
    const std::valarray<double>& predictRef(std::valarray<double> const& input);
    std::valarray<double> predict(std::valarray<double> const& input);
    void backpropagateSingle(const double& target, const double& learningRate, const double& positiveWeight = 1.0);
    void backpropagate(std::valarray<double> const& target, const double& learningRate, const double& positiveWeight = 1.0);
    void randomize(std::mt19937& random);
    void mutate(std::mt19937& random, const double& factor);
    void setLossType(LossFunction::Type type);
    LossFunction::Type getLossType() const;
    const char* getLossTypeName() const;

private:
    std::vector<Layer> layer;
    LossFunction lossFunction;
    bool initialized = false;
};

#endif