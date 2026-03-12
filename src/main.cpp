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
    // Select output loss mode here: BinaryCrossEntropy or MeanSquaredError.
    const LossFunction::Type selectedLossType = LossFunction::Type::BinaryCrossEntropy;
    network.setLossType(selectedLossType);

    // Upscaling factor for training samples (10x10 target -> 80x80 sampled grid).
    const int resolution = 8;
    // Effective training grid width/height after upscaling.
    const int dataX = static_cast<int>(smp[0].size()) * resolution;
    const int dataY = static_cast<int>(smp.size()) * resolution;
    // Half extents used to normalize x/y coordinates around the image center.
    const double dataXHalf = dataX / 2.0;
    const double dataYHalf = dataY / 2.0;
    // Total number of training points checked in one full pass.
    const double sampleCount = static_cast<double>(dataX * dataY);
    const int sampleCountInt = dataX * dataY;

    // Full training passes processed between UI refreshes.
    const int passesPerFrame = 32;
    // How strongly each correction changes the network at the start.
    const double initialLearningRate = 0.15;
    // Amount the correction strength shrinks after each full pass.
    const double learningRateDecay = 0.9995;
    // Smallest allowed correction strength, so learning does not stop.
    const double minimumLearningRate = 0.02;
    // Extra loss weight for positive pixels to preserve sparse smile dots.
    const double positiveClassWeight = 4.0;
    // Current correction strength, updated as training runs.
    double learningRate = initialLearningRate;

    std::vector<int> sampleOrder(sampleCountInt);
    std::iota(sampleOrder.begin(), sampleOrder.end(), 0);

    OutputWindow preview = OutputWindow(1000, 1000);
    int passes = 0;
    double startCost = 1.0;
    double cost = 0.0;

    while(preview.isOpen())
    {
        for(int passIndex = 0; passIndex < passesPerFrame; ++passIndex)
        {
            // Mix the point order each pass so the same scan pattern does not bias learning.
            std::shuffle(sampleOrder.begin(), sampleOrder.end(), random);
            cost = 0.0;
            for(int sampleIndex: sampleOrder)
            {
                const int x = sampleIndex / dataY;
                const int y = sampleIndex % dataY;
                std::valarray<double> output = network.predict({
                    (static_cast<double>(y) - dataYHalf) / dataY,
                    (static_cast<double>(x) - dataXHalf) / dataX
                });
                const double desired = smp[x / resolution][y / resolution];
                const double diff = output[0] - desired;
                cost += diff * diff;
                network.backpropagate({desired}, learningRate, positiveClassWeight);
            }
            cost /= sampleCount;
            // Decay learning rate over time to improve fine-detail convergence.
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
