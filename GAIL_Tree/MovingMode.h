#pragma once

#include "Models.h"

class MovingMode : public ModelNode {
public:
    MovingMode(int input_dim=10, int hidden_dim=32, int output_dim = 1) :
        ModelNode(input_dim, hidden_dim, output_dim) {}
    virtual void step(cppSimulatorImp* engine, bool default_action);
    virtual void train(const std::vector<PackedData>& data);
private:
};