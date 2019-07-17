#ifndef neuralnetwork_hpp
#define neuralnetwork_hpp

#include <random>
#include <vector>
#include <valarray>

class NeuralNetwork
{
public:
    struct Layer
    {
        std::valarray<double> value;
        std::valarray<double> bias;
        std::vector<
            std::valarray<double>
        > weight;
        Layer(int neurons, int links) : value(neurons), bias(neurons), weight(neurons, std::valarray<double>(links)){}        
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
    std::valarray<double> predict(std::valarray<double> const& input);
    void mutate(double factor);

private:
    std::vector<Layer> layer;
    bool initialized = false;
};

#endif