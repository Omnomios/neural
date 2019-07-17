#include <iostream>
#include "CostCalculator.hpp"

CostCalculator::CostCalculator()
{
    this->workThread = std::thread(&CostCalculator::workLoop, this);
}

CostCalculator::~CostCalculator() 
{
    this->running = false;
    this->workThread.join();
}


void CostCalculator::setData(std::vector<std::vector<double>> data)
{
    this->data = data;
}

void CostCalculator::addWork(NeuralNetwork* network)
{
    this->job.push(network);
}

void CostCalculator::start()
{
    while(!this->result.empty()) this->result.pop();
    this->done = false;
    this->condition.notify_all(); 
}

void CostCalculator::terminate()
{
    this->running = false;
    this->condition.notify_all(); 
}

void CostCalculator::workLoop() 
{
    std::unique_lock<std::mutex> lk(this->mutex);
    
    while(this->running)
    {
        this->working = false;
        this->condition.wait(lk);
        this->working = true;

        while(!this->job.empty())
        {
            NeuralNetwork* network = this->job.front();

            float resolution = 8;

            const float dataX = (float)this->data[0].size() * resolution;
            const float dataY = (float)this->data.size() * resolution;
            const float dataXh = dataX/2;
            const float dataYh = dataY/2;
            const float dataCount = dataX * dataY;

            double cost = 0;

            for(int x = 0; x < dataX; x++)
            for(int y = 0; y < dataY; y++)
            {
                double actual = network->predict({
                    ((float)y-dataYh) / dataY,
                    ((float)x-dataXh) / dataX
                })[0];
                
                double desired = this->data[x / resolution][y / resolution];
                cost += std::pow(actual - desired, 2);
            }

            cost /= dataCount;
            this->result.push({network, cost});
            this->job.pop();
        }
        this->done = true;
    }
    lk.unlock();
}
