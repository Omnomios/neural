#include <iostream>
#include <assert.h>
#include "NeuralNetwork.hpp"
#include "Matrix.hpp"

NeuralNetwork::NeuralNetwork() 
{
}

NeuralNetwork::NeuralNetwork(NeuralNetwork const& source) 
{
    this->set(source.dump());
}

NeuralNetwork::NeuralNetwork(std::vector<int> layers)
{
    std::vector<std::pair<int,int>> shapes;

    auto ita = layers.begin();
    auto itb = layers.begin();
    ita++;
    while(ita != layers.end())
    {
        //this->bias.push_back(Matrix::Zero(*ita,1));        
        shapes.push_back(std::pair<int,int>(*ita++,*itb++));
    } 

    for(auto const& value: shapes) 
    {
        //this->weight.push_back(Matrix::Zero(value.first, value.second));
    }
    this->initialized = true;
}

NeuralNetwork::~NeuralNetwork() 
{

}

std::pair<std::vector<Matrix>,std::vector<Matrix>> NeuralNetwork::dump() const
{
    return {this->weight, this->bias};
}

void NeuralNetwork::set(std::pair<std::vector<Matrix>,std::vector<Matrix>> data)
{
    this->weight = data.first;
    this->bias = data.second;
    this->initialized = true;
}

Matrix NeuralNetwork::predict(Matrix const& m) const
{
    assert(this->initialized);

    auto itw = this->weight.begin();
    auto itb = this->bias.begin();

    Matrix a = m;
    while(itw != this->weight.end() && itb != this->bias.end())
    {
        auto w = *itw;
        auto b = *itb;
        a = this->activation(this->matmul(w, a).col(0)+b);
        itw++;itb++;
    }

    return a;
}

void NeuralNetwork::mutate(double factor)
{
    assert(this->initialized);    

    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<> dist(-factor, factor);


    for(auto const& w: this->weight)
    {
        w += Matrix(w.rows(), w.cols()).random(-factor, factor);
    }

    for(auto const& b: this->bias)
    {
        for(long int i = 0; i < b.size(); i++)
        {
            double* v = const_cast<double*>(b.data()+i);
            *v += dist(e2);
        }
    }
}

Matrix NeuralNetwork::matmul(Matrix const& x, Matrix const& y) const
{
    Matrix result = Matrix::Zero(x.rows(), y.rows());

    // iterate through rows of X
    for(int i=0;i<x.rows();i++)
    {   // iterate through columns of Y
        for(int j=0;j<y.cols();j++)
        {   // iterate through rows of Y
            for(int k=0;k<y.rows();k++)
            {
                result(i,j) += x(i,k) * y(k,j);
            }
        }
    }

    return result;
}

Matrix NeuralNetwork::activation(Matrix const& m) const
{
    for(long int i = 0; i < m.size(); i++)
    {
        double* v = const_cast<double*>(m.data()+i);
        double value = *v;
        *v = 1.0f / (1.0f + std::exp(-value));
    }
    return m;
}