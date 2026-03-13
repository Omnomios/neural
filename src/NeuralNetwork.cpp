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


// Snapshot network state.
std::vector<NeuralNetwork::Layer> NeuralNetwork::dump () const
{
    return this->layer;
}
void NeuralNetwork::restore (std::vector<Layer> data)
{
    this->layer = data;
}


// Forward pass without copying output.
const std::valarray<double>& NeuralNetwork::predictRef (std::valarray<double> const& input)
{
    assert(this->initialized);

    if(input.size() != this->layer[0].value.size())
    {
        throw std::runtime_error("Input size doesn't match first layer in NeuralNetwork!");
    }

    // Load inputs into layer 0.
    this->layer[0].value = input;

    // Walk each layer pair and compute next activations.
    for(std::size_t i = 0; i < this->layer.size()-1; ++i)
    {
        Layer& sourceLayer = this->layer[i];
        Layer& destLayer = this->layer[i+1];
        const std::size_t sourceCount = sourceLayer.weight.size();
        const std::size_t destCount = destLayer.value.size();

        // Start from bias.
        for(std::size_t dst = 0; dst < destCount; ++dst)
        {
            destLayer.value[dst] = destLayer.bias[dst];
        }

        // Add weighted source contributions (explicit loops avoid temp objects).
        for(std::size_t src = 0; src < sourceCount; ++src)
        {
            const double sourceValue = sourceLayer.value[src];
            const std::valarray<double>& weightRow = sourceLayer.weight[src];
            for(std::size_t dst = 0; dst < destCount; ++dst)
            {
                destLayer.value[dst] += sourceValue * weightRow[dst];
            }
        }

        // Sigmoid in-place.
        for(std::size_t dst = 0; dst < destCount; ++dst)
        {
            const double summed = destLayer.value[dst];
            destLayer.value[dst] = 1.0 / (1.0 + std::exp(-summed));
        }
    }

    return this->layer.back().value;
}

std::valarray<double> NeuralNetwork::predict (std::valarray<double> const& input)
{
    return this->predictRef(input);
}

void NeuralNetwork::backpropagateSingle (const double& target, const double& learningRate, const double& positiveWeight)
{
    assert(this->initialized);

    Layer& output = this->layer.back();
    if(output.delta.size() != 1)
    {
        throw std::runtime_error("backpropagateSingle requires a single output neuron!");
    }

    // Single-output delta.
    output.delta[0] = this->lossFunction.outputDelta(output.value[0], target, positiveWeight);

    // Backprop hidden deltas.
    for(std::size_t layerIndex = this->layer.size() - 1; layerIndex > 1; --layerIndex)
    {
        Layer& current = this->layer[layerIndex - 1];
        Layer& next = this->layer[layerIndex];
        const std::size_t currentCount = current.weight.size();
        const std::size_t nextCount = next.delta.size();

        for(std::size_t neuron = 0; neuron < currentCount; ++neuron)
        {
            // Pull error signal from the next layer.
            double weightedError = 0.0;
            const std::valarray<double>& weightRow = current.weight[neuron];
            for(std::size_t nextNeuron = 0; nextNeuron < nextCount; ++nextNeuron)
            {
                weightedError += weightRow[nextNeuron] * next.delta[nextNeuron];
            }
            // Sigmoid slope from current activation.
            const double activation = current.value[neuron];
            current.delta[neuron] = weightedError * activation * (1.0 - activation);
        }
    }

    // Apply gradient step to weights and biases.
    for(std::size_t layerIndex = 0; layerIndex < this->layer.size() - 1; ++layerIndex)
    {
        Layer& source = this->layer[layerIndex];
        Layer& dest = this->layer[layerIndex + 1];
        const std::size_t sourceCount = source.weight.size();
        const std::size_t destCount = dest.delta.size();

        for(std::size_t src = 0; src < sourceCount; ++src)
        {
            const double scale = learningRate * source.value[src];
            std::valarray<double>& weightRow = source.weight[src];
            for(std::size_t dst = 0; dst < destCount; ++dst)
            {
                weightRow[dst] -= scale * dest.delta[dst];
            }
        }

        for(std::size_t dst = 0; dst < destCount; ++dst)
        {
            dest.bias[dst] -= learningRate * dest.delta[dst];
        }
    }
}

void NeuralNetwork::backpropagate (std::valarray<double> const& target, const double& learningRate, const double& positiveWeight)
{
    assert(this->initialized);

    if(target.size() != this->layer.back().value.size())
    {
        throw std::runtime_error("Target size doesn't match output layer in NeuralNetwork!");
    }

    // Output deltas from selected loss.
    Layer& output = this->layer.back();
    const std::size_t outputCount = output.delta.size();
    for(std::size_t neuron = 0; neuron < outputCount; ++neuron)
    {
        output.delta[neuron] = this->lossFunction.outputDelta(
            output.value[neuron],
            target[neuron],
            positiveWeight
        );
    }

    // Backprop hidden deltas.
    for(std::size_t layerIndex = this->layer.size() - 1; layerIndex > 1; --layerIndex)
    {
        Layer& current = this->layer[layerIndex - 1];
        Layer& next = this->layer[layerIndex];

        const std::size_t currentCount = current.weight.size();
        const std::size_t nextCount = next.delta.size();
        for(std::size_t neuron = 0; neuron < currentCount; ++neuron)
        {
            // Pull error signal from the next layer.
            double weightedError = 0.0;
            const std::valarray<double>& weightRow = current.weight[neuron];
            for(std::size_t nextNeuron = 0; nextNeuron < nextCount; ++nextNeuron)
            {
                weightedError += weightRow[nextNeuron] * next.delta[nextNeuron];
            }
            // Sigmoid slope from current activation.
            const double activation = current.value[neuron];
            current.delta[neuron] = weightedError * activation * (1.0 - activation);
        }
    }

    // Apply gradient step to weights and biases.
    for(std::size_t layerIndex = 0; layerIndex < this->layer.size() - 1; ++layerIndex)
    {
        Layer& source = this->layer[layerIndex];
        Layer& dest = this->layer[layerIndex + 1];
        const std::size_t sourceCount = source.weight.size();
        const std::size_t destCount = dest.delta.size();

        for(std::size_t src = 0; src < sourceCount; ++src)
        {
            const double scale = learningRate * source.value[src];
            std::valarray<double>& weightRow = source.weight[src];
            for(std::size_t dst = 0; dst < destCount; ++dst)
            {
                weightRow[dst] -= scale * dest.delta[dst];
            }
        }

        for(std::size_t dst = 0; dst < destCount; ++dst)
        {
            dest.bias[dst] -= learningRate * dest.delta[dst];
        }
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
