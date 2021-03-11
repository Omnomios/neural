#include <iostream>
#include <assert.h>
#include <valarray>
#include <exception>

#include "NeuralNetwork.hpp"

NeuralNetwork::NeuralNetwork ()
{
}

NeuralNetwork::NeuralNetwork (NeuralNetwork const& rhs)
{
    this->restore(rhs.dump());
    this->initialized = true;
}

NeuralNetwork::NeuralNetwork (std::vector<int> layers)
{
    this->layer.clear();

    auto linkedLayer = layers.begin();
    auto currentLayer = layers.begin();
    linkedLayer++;
    while(currentLayer != layers.end())
    {
        // Use the struct constructor to build the data for the layer
        this->layer.push_back(Layer(*currentLayer,(linkedLayer == layers.end())?0:*linkedLayer));

        currentLayer++;
        linkedLayer++;
    }

    this->initialized = true;
}
NeuralNetwork::~NeuralNetwork () { }


/*
* Saves and restores the state of the neurons
 */
std::vector<NeuralNetwork::Layer> NeuralNetwork::dump () const
{
    return this->layer;
}
void NeuralNetwork::restore (std::vector<Layer> data)
{
    this->layer = data;
}


/*
* Takes a valarray of doubles that matches the size of the first layer and
* return the result of the network calculation.
 */
std::valarray<double> NeuralNetwork::predict (std::valarray<double> const& input)
{
    assert(this->initialized);

    if(input.size() != this->layer[0].value.size())
    {
        throw std::runtime_error("Input size doesn't match first layer in NeuralNetwork!");
    }

    // Set the input layer on the network
    this->layer[0].value = input;

    for(unsigned int i = 0; i < this->layer.size()-1; i++)
    {
        Layer& sourceLayer = this->layer[i];
        Layer& destLayer = this->layer[i+1];

        // Reset target neurons
        destLayer.value = 0;

        // Add up weighted inputs and apply them to the target
        for(unsigned int neuron = 0; neuron < sourceLayer.weight.size(); neuron++)
        {
            destLayer.value += sourceLayer.value[neuron] * sourceLayer.weight[neuron];
        }

        // Apply the bias values
        destLayer.value += destLayer.bias;

        // Activation function
        destLayer.value = 1.0 / (1.0 + std::exp(-(destLayer.value)));
    }

    return this->layer.back().value;
}

/*
* Applies a random number to every neuron and bias in the network.
*
* Note: used for random network evolution.
*/

void NeuralNetwork::mutate (std::mt19937& random, const double& factor)
{
    assert(this->initialized);
    std::uniform_real_distribution<> dist(-factor, factor);

    for(auto& layer: this->layer)
    {
        layer.bias += dist(random);
        for(auto& weight: layer.weight)
        {
            for(double& value: weight)
            {
                value += dist(random);
            }
        }
    }
}
