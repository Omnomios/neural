#ifndef outputWindow_hpp
#define outputWindow_hpp

#include <thread>
#include <mutex>
#include <condition_variable>
#include <SFML/Graphics.hpp>
#include "NeuralNetwork.hpp"

class OutputWindow
{
public:
    OutputWindow(int x, int y);
    ~OutputWindow();

    void renderLoop();
    void messageLoop();
    
    void showNetwork(NeuralNetwork nn, int resolution);
    void terminate();

private:
    sf::RenderWindow window;

    std::thread renderThread;
    std::thread messageThread;

    NeuralNetwork network;
        
    const int windowX = 1000;
    const int windowY = 1000;

    int resolution = 100;

    bool running = true;
    bool working = false;

    std::mutex mutex;
    std::condition_variable condition;
};

#endif