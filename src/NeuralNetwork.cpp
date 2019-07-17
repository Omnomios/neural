#include <iostream>
#include <assert.h>
#include <valarray>
#include "NeuralNetwork.hpp"

NeuralNetwork::NeuralNetwork() 
{
}

NeuralNetwork::NeuralNetwork(NeuralNetwork const& rhs)
{
    this->restore(rhs.dump());
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
        this->layer.push_back({
            std::valarray<double>(*itb),        // value
            std::valarray<double>(*itb),        // bias
            std::vector<std::valarray<double>>(    // weights
                *itb,
                std::valarray<double>((ita == layers.end())?0:*ita)
            )
        });
        ita++;itb++;
    } 

    this->initialized = true;
}

NeuralNetwork::~NeuralNetwork() 
{

}

std::vector<NeuralNetwork::Layer> NeuralNetwork::dump() const
{
    return this->layer;
}
void NeuralNetwork::restore(std::vector<Layer> data)
{
    this->layer = data;
}

std::valarray<double> NeuralNetwork::predict(std::valarray<double> const& input)
{
    assert(this->initialized);

    this->layer[0].value = input;

    for(unsigned int i = 0; i < this->layer.size()-1; i++)
    {
        Layer& sourceLayer = this->layer[i];
        Layer& destLayer = this->layer[i+1];

        // Reset
        destLayer.value -= destLayer.value;

        // Add up weighted inputs
        for(unsigned int neuron = 0; neuron < sourceLayer.weight.size(); neuron++)
        {            
            destLayer.value += sourceLayer.value[neuron] * sourceLayer.weight[neuron];
        }

        // Need to reset the registers so they don't persist over predicitons.
        //sourceLayer.value -= sourceLayer.value;
        
        // Apply the bias values
        destLayer.value += destLayer.bias;

        // Activation function
        destLayer.value = 1.0/(1.0 + std::exp(-(destLayer.value)));
    } 

    return this->layer.back().value;
}

void NeuralNetwork::mutate(double factor)
{
    assert(this->initialized);    
    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<> dist(-factor, factor);

    for(auto& layer: this->layer)
    {
        layer.bias += dist(e2);
        for(auto& weight: layer.weight)
        {
            for(double& value: weight)
            {
                value += dist(e2);
            }
        }
    }
}
