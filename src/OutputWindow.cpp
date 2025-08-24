#include "OutputWindow.hpp"

#include "hackttf.h"
#include <optional>

OutputWindow::OutputWindow(int x, int y): windowX(x), windowY(y)
{
    this->running = true;
    this->renderThread = std::thread(&OutputWindow::renderLoop, this);
}

OutputWindow::~OutputWindow()
{
    this->running = false;
    this->renderThread.join();
}

void OutputWindow::showNetwork(const NeuralNetwork& nn, int resolution, const std::string& info)
{
    if(this->working) return; // Discard new network if rendering
    this->resolution = resolution;
    this->network = nn;
    this->infoText = info;
    if (this->text)
    {
        this->text->setString(this->infoText);
    }
    this->condition.notify_all();
}

void OutputWindow::terminate()
{
    this->running = false;
    this->condition.notify_all();
}

void OutputWindow::renderLoop()
{
    std::unique_lock<std::mutex> lk(this->mutex);

    if (!this->font.openFromMemory(hack_ttf, hack_ttf_len))
    {
        this->running = false;
        return;
    }
    this->text.emplace(this->font, "", 24);
    this->text->setFillColor(sf::Color::White);

    this->window.create(sf::VideoMode({this->windowX, this->windowY}), "Output");

    while(this->running)
    {
        this->working = false;
        this->condition.wait(lk);
        this->working = true;

        if(this->window.isOpen())
        {
            this->window.clear(sf::Color::Black);

            sf::RectangleShape rectangle(sf::Vector2f(this->resolution, this->resolution));

            for(int x = 0; x < this->windowX; x+=this->resolution)
            for(int y = 0; y < this->windowY; y+=this->resolution)
            {
                double actual = this->network.predict({
                    ((float)x-(this->windowX/2)) / this->windowX,
                    ((float)y-(this->windowY/2)) / this->windowY
                })[0];
                rectangle.setPosition({(float)x,(float)y});
                rectangle.setFillColor(sf::Color(100*actual, 255*actual, 20*actual));
                this->window.draw(rectangle);
            }


            if (this->text)
            {
                this->text->setString(this->infoText);
                this->text->setPosition({0.f,0.f});
                window.draw(*this->text);
            }

            this->window.display();

            while (const std::optional<sf::Event> event = this->window.pollEvent())
            {
                if (event->is<sf::Event::Closed>())
                {
                    this->window.close();
                    this->terminate();
                }
            }
        }
    }
    lk.unlock();
}