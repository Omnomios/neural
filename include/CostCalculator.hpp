#ifndef costCalculator_hpp
#define costCalculator_hpp

#include <thread>
#include <mutex>
#include <condition_variable>

#include <queue>
#include "NeuralNetwork.hpp"
#include "Matrix.hpp"

class CostCalculator
{
public:
    CostCalculator();
    ~CostCalculator();

    void workLoop();    
    void setData(Matrix data);
    void addWork(NeuralNetwork* network);
    void start();
    void terminate();

    bool done = true;
    std::queue<std::pair<NeuralNetwork*, double>> result;

private:
    std::thread workThread;

    std::queue<NeuralNetwork*> job;

    Matrix data;

    bool running = true;
    bool working = false;

    std::mutex mutex;
    std::condition_variable condition;
};

#endif