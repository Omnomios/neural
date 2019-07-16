#ifndef neuralnetwork_hpp
#define neuralnetwork_hpp

#include <random>
#include <vector>
#include <valarray>

typedef std::valarray<double> vecXd;

struct neuron
{
    double value;
    double bias;
    std::valarray<double> weight;
};


class NeuralNetwork
{
public:
    NeuralNetwork();
    NeuralNetwork(NeuralNetwork const& rhs);
    NeuralNetwork(std::vector<int> layers);
    ~NeuralNetwork();

    std::vector<std::vector<neuron>> get() const;
    void set(std::vector<std::vector<neuron>>);

    std::valarray<double> predict(std::valarray<double> const& input);
    void mutate(double factor);

private:
    std::vector<std::vector<neuron>> layer;
    bool initialized = false;
};

#endif