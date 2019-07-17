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
        std::vector<std::valarray<double>> weight;
    };

public:
    NeuralNetwork();
    NeuralNetwork(NeuralNetwork const& rhs);
    NeuralNetwork(std::vector<int> layers);
    ~NeuralNetwork();

    std::vector<Layer> dump() const;
    void restore(std::vector<Layer>);

    std::valarray<double> predict(std::valarray<double> const& input);
    void mutate(double factor);

    std::vector<Layer> layer;
private:
    bool initialized = false;
};

#endif