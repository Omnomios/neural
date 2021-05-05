#include <iostream>
#include <vector>
#include <map>
#include <limits>
#include <chrono>

#ifdef WIN32
#include <windows.h>
#endif

#include "NeuralNetwork.hpp"
#include "OutputWindow.hpp"
#include "CostCalculator.hpp"

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

    /*
    std::vector<std::vector<double>> smp(
        {
            {0,1,0,1,0,1,0,1,0,1},
            {1,0,1,0,1,0,1,0,1,0},
            {0,1,0,1,0,1,0,1,0,1},
            {1,0,1,0,1,0,1,0,1,0},
            {0,1,0,1,0,1,0,1,0,1},
            {1,0,1,0,1,0,1,0,1,0},
            {0,1,0,1,0,1,0,1,0,1},
            {1,0,1,0,1,0,1,0,1,0},
            {0,1,0,1,0,1,0,1,0,1},
            {1,0,1,0,1,0,1,0,1,0}
        }
    );
    */
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

    std::vector<NeuralNetwork> networks(48);
    for(NeuralNetwork& network: networks)
    {
        network = NeuralNetwork({2, 16, 16, 1});
    }

    std::random_device rd;
    std::mt19937 e2(rd());
    e2.seed(time());

    const int threads = 12;
    std::vector<CostCalculator> calculate(threads);
    for(CostCalculator& worker: calculate) worker.setData(smp);

    std::pair<double, NeuralNetwork> bestCost;
    bestCost.first = std::numeric_limits<double>::max();
    bestCost.second = networks[0];

    OutputWindow preview = OutputWindow(1000, 1000);
    bool waiting = false;

    int generation = 0;
    double startCost = 1.0;

    while(preview.isOpen())
    {
        // Get a benchmark to calculate effectiveness of training.
        if(generation == 1) startCost = bestCost.first;

        double baseCost = bestCost.first;
        bestCost.first = std::numeric_limits<double>::max();
        int index = 0;


        // Get the best child from the current generation.
        for(NeuralNetwork& network: networks)
        {
            // Breed the chosen one
            network = bestCost.second;
            network.mutate(e2, std::min(1.0, std::max(baseCost, 0.001)));
            calculate[index%threads].addWork(&network);
            index++;
        }
        //std::this_thread::sleep_for(std::chrono::milliseconds(1000));

        for(CostCalculator& worker: calculate) worker.start();

        int confidence = (100-std::round((baseCost / startCost)*100));

        // Update the outputs while we're waiting for the result.
        preview.showNetwork(
            bestCost.second,
            std::max((int)(baseCost*500), 10),
            "Generation: " + std::to_string(generation++) + "\nCost: " + std::to_string(baseCost) + "\nConfidence: " + std::to_string(confidence) + "%\n"
        );

        // Get the result.
        waiting = true;
        while(waiting)
        {
            int results = 0;
            for(CostCalculator& worker: calculate)
            {
                if(worker.done) results++;
            }

            if(results == threads)
            {
                waiting = false; // break out of loop
                for(CostCalculator& worker: calculate)
                {
                    while(!worker.result.empty())
                    {
                        std::pair<NeuralNetwork*, double> result = worker.result.front();
                        worker.result.pop();

                        if(bestCost.first > result.second)
                        {
                            bestCost.first = result.second;
                            bestCost.second = *result.first;
                        }
                    }
                }

                break;
            }
            std::this_thread::yield();
        }

    }

    for(CostCalculator& worker: calculate) worker.terminate();

    return 0;
}
