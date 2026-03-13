#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <numeric>

#ifdef WIN32
#include <windows.h>
#endif

#include "NeuralNetwork.hpp"
#include "OutputWindow.hpp"

int64_t time()
{
    std::chrono::milliseconds ms = std::chrono::duration_cast< std::chrono::milliseconds >(
        std::chrono::system_clock::now().time_since_epoch()
    );
    return ms.count();
}

#ifdef WIN32
int WINAPI WinMain (HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nShowCmd)
#else
int main(int argc, char *argv[])
#endif
{
    std::vector<std::vector<double>> smp(
        {
            {0,0,0,0,0,0,0,0,0,0},
            {0,0,0,1,0,0,1,0,0,0},
            {0,0,0,1,0,0,1,0,0,0},
            {0,0,0,1,0,0,1,0,0,0},
            {0,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0},
            {0,1,0,0,0,0,0,0,1,0},
            {0,0,1,0,0,0,0,1,0,0},
            {0,0,0,1,1,1,1,0,0,0},
            {0,0,0,0,0,0,0,0,0,0}
        }
    );

    std::random_device rd;
    std::mt19937 random(rd());
    random.seed(time());

    NeuralNetwork network({2, 24, 24, 1});
    network.randomize(random);
    // Choose output loss mode.
    const LossFunction::Type selectedLossType = LossFunction::Type::BinaryCrossEntropy;
    network.setLossType(selectedLossType);

    // Upscale the 10x10 target grid for denser training points.
    const int resolution = 8;
    // Training grid size after scaling.
    const int dataX = static_cast<int>(smp[0].size()) * resolution;
    const int dataY = static_cast<int>(smp.size()) * resolution;
    // Center offsets for normalized input coordinates.
    const double dataXHalf = dataX / 2.0;
    const double dataYHalf = dataY / 2.0;
    // Points processed per pass.
    const double sampleCount = static_cast<double>(dataX * dataY);
    const int sampleCountInt = dataX * dataY;

    // Passes to run before each UI refresh.
    const int passesPerFrame = 32;
    // Starting update scale.
    const double initialLearningRate = 0.15;
    // Per-pass learning-rate decay.
    const double learningRateDecay = 0.9995;
    // Learning-rate floor.
    const double minimumLearningRate = 0.02;
    // Weight positive pixels higher to keep sparse dots.
    const double positiveClassWeight = 4.0;
    // Current learning rate.
    double learningRate = initialLearningRate;

    std::vector<int> sampleOrder(sampleCountInt);
    std::iota(sampleOrder.begin(), sampleOrder.end(), 0);
    std::valarray<double> input(2);

    OutputWindow preview = OutputWindow(1000, 1000);
    int passes = 0;
    double startCost = 1.0;
    double cost = 0.0;

    while(preview.isOpen())
    {
        for(int passIndex = 0; passIndex < passesPerFrame; ++passIndex)
        {
            // Shuffle sample order each pass.
            std::shuffle(sampleOrder.begin(), sampleOrder.end(), random);
            cost = 0.0;
            for(int sampleIndex: sampleOrder)
            {
                const int x = sampleIndex / dataY;
                const int y = sampleIndex % dataY;
                input[0] = (static_cast<double>(y) - dataYHalf) / dataY;
                input[1] = (static_cast<double>(x) - dataXHalf) / dataX;
                const std::valarray<double>& output = network.predictRef(input);
                const double desired = smp[x / resolution][y / resolution];
                const double diff = output[0] - desired;
                cost += diff * diff;
                network.backpropagateSingle(desired, learningRate, positiveClassWeight);
            }
            cost /= sampleCount;
            // Decay learning rate each pass.
            learningRate = std::max(minimumLearningRate, learningRate * learningRateDecay);
            passes++;
        }

        if(passes == passesPerFrame)
        {
            startCost = cost;
        }

        int confidence = 0;
        if(startCost > 0.0)
        {
            confidence = static_cast<int>(100 - std::round((cost / startCost) * 100.0));
        }
        confidence = std::clamp(confidence, 0, 100);

        preview.showNetwork(
            network,
            std::max(static_cast<int>(cost * 500), 10),
            "Pass: " + std::to_string(passes) + "\nLoss: " + network.getLossTypeName() + "\nCost: " + std::to_string(cost) + "\nLR: " + std::to_string(learningRate) + "\nConfidence: " + std::to_string(confidence) + "%\n"
        );
    }

    return 0;
}
