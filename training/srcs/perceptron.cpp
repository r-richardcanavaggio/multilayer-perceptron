#include "perceptron.hpp"
#include <cmath>
#include <iostream>

double  sigmoid( double x )
{
    return 1 / (1 + std::exp(-x));
}

int main( void )
{
    std::vector<double> weights = {0.05, 0.00, -0.47};
    double              bias = 0.25;

    Perceptron          perceptron(weights, bias);
    std::vector<double> inputs = {1.0, 2.0, 3.0};

    double              output = perceptron.forward(inputs, sigmoid);

    std::cout << "output: " << output << "\n";
    return (0);
}