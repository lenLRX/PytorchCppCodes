#pragma once

#include "Models.h"

class LastHitMode : public ModelNode {
public:
    LastHitMode(int input_dim, int hidden_dim, int output_dim = 1) :
        ModelNode(input_dim, hidden_dim, output_dim) {}

    virtual void step(cppSimulatorImp* engine, bool default_action);

private:
    std::vector<int> mask;//mask invalid ticks
};