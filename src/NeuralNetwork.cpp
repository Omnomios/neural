#include <iostream>
#include <assert.h>
#include <valarray>
#include "NeuralNetwork.hpp"

NeuralNetwork::NeuralNetwork() 
{
}

NeuralNetwork::NeuralNetwork(NeuralNetwork const& rhs)
{
    this->set(rhs.get());
    this->initialized = true;
}

NeuralNetwork::NeuralNetwork(std::vector<int> layers)
{
    this->layer.clear();

    auto ita = layers.begin();
    auto itb = layers.begin();
    ita++;
    while(itb != layers.end())
    {
        this->layer.push_back(std::vector<neuron>(*itb, {0, 0, std::valarray<double>((ita == layers.end())?0:*ita)}));        
        ita++;itb++;
    } 

    this->initialized = true;
}

NeuralNetwork::~NeuralNetwork() 
{

}

std::vector<std::vector<neuron>> NeuralNetwork::get() const
{
    return this->layer;
}
void NeuralNetwork::set(std::vector<std::vector<neuron>> data)
{
    this->layer = data;
}

std::valarray<double> NeuralNetwork::predict(std::valarray<double> const& input)
{
    assert(this->initialized);

    std::valarray<double> a(input);

    for(unsigned int i = 0; i < this->layer.size()-1; i++)
    {
        auto& sourceLayer = this->layer[i];
        auto& destLayer = this->layer[i+1];

        // TODO: optimise?
        std::valarray<double> r(destLayer.size());

        // Add up weighted inputs
        for(unsigned int neuron = 0; neuron < sourceLayer.size(); neuron++)
        {
            r += a[neuron] * sourceLayer[neuron].weight;
        }

        // Apply bias
        for(unsigned int neuron = 0; neuron < destLayer.size(); neuron++)
        {
            r[neuron] += destLayer[neuron].bias;
        }

        // Activation function
        a = 1.0/(1.0 + std::exp(-r));
    } 

    return a;
}

void NeuralNetwork::mutate(double factor)
{
    assert(this->initialized);    
    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<> dist(-factor, factor);

    for(auto& layer: this->layer)
    {
        for(auto& neuron: layer)
        {
            neuron.bias += dist(e2);
            for(double& value: neuron.weight)
            {
                value += dist(e2);
            }
        }
    }
}
