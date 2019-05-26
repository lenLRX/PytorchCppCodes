#pragma once

#include "Models.h"

class ActionChoice : public ModelNode {
public:
    ActionChoice(int input_dim=10, int hidden_dim=32, int output_dim = 2);

    virtual const std::string& type() { return "ActionChoice"; }

    virtual void step(cppSimulatorImp* engine, bool default_action);

    virtual void train(const std::vector<PackedData>& data);

private:
    int get_default(torch::Tensor x);
    
};