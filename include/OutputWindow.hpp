#ifndef outputWindow_hpp
#define outputWindow_hpp

#include <thread>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <string>
#include <SFML/Graphics.hpp>
#include "NeuralNetwork.hpp"

class OutputWindow
{
public:
    OutputWindow(int x, int y);
    ~OutputWindow();

    void renderLoop();

    void showNetwork(const NeuralNetwork& nn, int resolution, const std::string& info);
    void terminate();

    bool isOpen() { return this->running; };

private:
    sf::RenderWindow window;
    sf::Font font;
    std::optional<sf::Text> text;

    std::thread renderThread;

    NeuralNetwork network;

    const int windowX = 1000;
    const int windowY = 1000;

    int resolution = 100;
    std::string infoText;

    bool running = true;
    bool working = false;

    std::mutex mutex;
    std::condition_variable condition;
};

#endif