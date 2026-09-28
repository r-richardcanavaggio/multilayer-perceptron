#pragma once

#include <vector>
#include <stdexcept>

class Perceptron
{
    private:
        std::vector<double> weights;
        double              bias;

    public:
        Perceptron( const std::vector<double>& initW, double initB ) : weights(initW), bias(initB) {}

        double forward( const std::vector<double>& inputs, double (*activationFunc)(double))
        {
            if (inputs.size() != weights.size())
                throw std::invalid_argument("Inputs size must match weights size");
            
            double weightedSum = 0.0;

            for (std::size_t i = 0; i < inputs.size(); i++)
                weightedSum += inputs[i] * weights[i];
            
            weightedSum += bias;
            return activationFunc(weightedSum);
        }
};