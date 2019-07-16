#include <iostream>
#include <vector>
#include <map>
#include <limits>
#include <chrono>

#include "NeuralNetwork.hpp"
#include "OutputWindow.hpp"
#include "CostCalculator.hpp"

void time()
{
    std::chrono::milliseconds ms = std::chrono::duration_cast< std::chrono::milliseconds >(
        std::chrono::system_clock::now().time_since_epoch()
    );

    //std::cout << (int)ms << std::endl;
}


int main() 
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


    std::vector<NeuralNetwork> networks(16);
    for(NeuralNetwork& network: networks) network = NeuralNetwork({2, 16, 16, 1});

    const int threads = 1;
    std::vector<CostCalculator> calculate(threads);
    for(CostCalculator& worker: calculate) worker.setData(smp);

    std::pair<double, NeuralNetwork> bestCost;
    bestCost.first = std::numeric_limits<double>::max();
    bestCost.second = networks[0];

    OutputWindow preview = OutputWindow(1000, 1000);
    bool waiting = false;

    int generation = 0;

    while(true)
    {
        // Get the best child from the current generation.
        double basecost = bestCost.first;
        bestCost.first = std::numeric_limits<double>::max();
        int index = 0;
        
        for(NeuralNetwork& network: networks)
        {
            // Breed the chosen one
            network = bestCost.second;
            network.mutate(std::min(0.5, std::max(basecost, 0.001)));
            calculate[index%threads].addWork(&network);
            index++;
        }
        //std::this_thread::sleep_for(std::chrono::milliseconds(1000));

        for(CostCalculator& worker: calculate) worker.start();

        // Update the outputs while we're waiting for the result.
        preview.showNetwork(bestCost.second, std::max((int)(basecost*500), 10));
        std::cout << "\e[1;1H\e[2J" << std::endl;
        std::cout << "Generation: "<< generation++ << "\nError: " << basecost << "\n";

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

    return 0;
}
