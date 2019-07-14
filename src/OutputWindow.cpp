#include "OutputWindow.hpp"

OutputWindow::OutputWindow(int x, int y): windowX(x), windowY(y)
{
    this->renderThread = std::thread(&OutputWindow::renderLoop, this);
    this->messageThread = std::thread(&OutputWindow::messageLoop, this);
}

OutputWindow::~OutputWindow() 
{
    this->running = false;
    this->renderThread.join();
    this->messageThread.join();
}

void OutputWindow::showNetwork(NeuralNetwork nn, int resolution)
{
    if(this->working) return; // Discard new network if rendering
    this->resolution = resolution;
    this->network = nn;
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
    
    this->window.create(sf::VideoMode(this->windowX, this->windowY), "Output");

    Matrix m = Matrix::Zero(2,1);

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
                m(0) = ((float)x-(this->windowX/2)) / this->windowX;
                m(1) = ((float)y-(this->windowY/2)) / this->windowY;

                double actual = this->network.predict(m)(0);
                rectangle.setPosition(x,y);
                rectangle.setFillColor(sf::Color(255*actual, 255*actual, 255*actual));
                this->window.draw(rectangle);
            }

            this->window.display();
        }
    }
    lk.unlock();
}

void OutputWindow::messageLoop() 
{
    while(this->running)
    {
        sf::Event event;
        while (this->window.pollEvent(event))
        {
            switch (event.type)
            {
                case sf::Event::Closed:
                    this->window.close();
                    this->terminate();
                break;
                default:
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}
