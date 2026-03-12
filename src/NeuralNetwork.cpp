#include <iostream>
#include <assert.h>
#include <valarray>
#include <exception>
#include <cmath>

#include "NeuralNetwork.hpp"

NeuralNetwork::NeuralNetwork ()
{
}

NeuralNetwork::NeuralNetwork (NeuralNetwork const& rhs)
{
    this->restore(rhs.dump());
    this->lossFunction = rhs.lossFunction;
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

void NeuralNetwork::backpropagate (std::valarray<double> const& target, const double& learningRate, const double& positiveWeight)
{
    assert(this->initialized);

    if(target.size() != this->layer.back().value.size())
    {
        throw std::runtime_error("Target size doesn't match output layer in NeuralNetwork!");
    }

    Layer& output = this->layer.back();
    for(std::size_t neuron = 0; neuron < output.delta.size(); ++neuron)
    {
        // Delegate output-layer loss gradient calculation to selected loss mode.
        output.delta[neuron] = this->lossFunction.outputDelta(
            output.value[neuron],
            target[neuron],
            positiveWeight
        );
    }

    for(std::size_t layerIndex = this->layer.size() - 1; layerIndex > 1; --layerIndex)
    {
        Layer& current = this->layer[layerIndex - 1];
        Layer& next = this->layer[layerIndex];

        current.delta = 0.0;
        for(std::size_t neuron = 0; neuron < current.weight.size(); ++neuron)
        {
            current.delta[neuron] = (current.weight[neuron] * next.delta).sum();
        }

        current.delta *= current.value * (1.0 - current.value);
    }

    for(std::size_t layerIndex = 0; layerIndex < this->layer.size() - 1; ++layerIndex)
    {
        Layer& source = this->layer[layerIndex];
        Layer& dest = this->layer[layerIndex + 1];

        for(std::size_t neuron = 0; neuron < source.weight.size(); ++neuron)
        {
            source.weight[neuron] -= dest.delta * (learningRate * source.value[neuron]);
        }
        dest.bias -= learningRate * dest.delta;
    }
}

void NeuralNetwork::randomize (std::mt19937& random)
{
    assert(this->initialized);

    for(std::size_t layerIndex = 0; layerIndex < this->layer.size() - 1; ++layerIndex)
    {
        Layer& source = this->layer[layerIndex];
        Layer& dest = this->layer[layerIndex + 1];

        const double fanIn = static_cast<double>(source.value.size());
        const double fanOut = static_cast<double>(dest.value.size());
        const double stdDev = std::sqrt(2.0 / (fanIn + fanOut));
        std::normal_distribution<double> dist(0.0, stdDev);

        for(double& bias: dest.bias)
        {
            bias = dist(random);
        }

        for(auto& neuronWeights: source.weight)
        {
            for(double& weight: neuronWeights)
            {
                weight = dist(random);
            }
        }
    }
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

void NeuralNetwork::setLossType(LossFunction::Type type)
{
    this->lossFunction.setType(type);
}

LossFunction::Type NeuralNetwork::getLossType() const
{
    return this->lossFunction.getType();
}

const char* NeuralNetwork::getLossTypeName() const
{
    return this->lossFunction.getTypeName();
}
