#include <cmath>
#include "LossFunction.hpp"

LossFunction::LossFunction(Type type): type(type)
{
}

void LossFunction::setType(Type type)
{
    this->type = type;
}

LossFunction::Type LossFunction::getType() const
{
    return this->type;
}

const char* LossFunction::getTypeName() const
{
    switch(this->type)
    {
        case Type::MeanSquaredError:
            return "MSE";
        case Type::BinaryCrossEntropy:
            return "BCE";
        default:
            return "Unknown";
    }
}

double LossFunction::outputDelta(double prediction, double target, double positiveWeight) const
{
    const double classWeight = (target > 0.5) ? positiveWeight : 1.0;

    switch(this->type)
    {
        case Type::MeanSquaredError:
            // MSE + sigmoid output derivative.
            return (prediction - target) * prediction * (1.0 - prediction) * classWeight;

        case Type::BinaryCrossEntropy:
            // BCE + sigmoid output derivative simplifies to (prediction - target).
            return (prediction - target) * classWeight;

        default:
            return (prediction - target) * classWeight;
    }
}
