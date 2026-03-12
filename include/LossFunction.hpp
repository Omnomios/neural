#ifndef lossfunction_hpp
#define lossfunction_hpp

class LossFunction
{
public:
    enum class Type
    {
        MeanSquaredError,
        BinaryCrossEntropy
    };

public:
    LossFunction(Type type = Type::BinaryCrossEntropy);
    void setType(Type type);
    Type getType() const;
    const char* getTypeName() const;
    double outputDelta(double prediction, double target, double positiveWeight = 1.0) const;

private:
    Type type;
};

#endif
