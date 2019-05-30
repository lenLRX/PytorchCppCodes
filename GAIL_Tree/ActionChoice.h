#pragma once

#include "Models.h"

class ActionChoice : public ModelNode {
public:
    explicit ActionChoice(int input_dim=10, int hidden_dim=32, int output_dim = 2);

    const std::string& type() override {
        static const std::string type_name = "ActionChoice";
        return type_name;
    }

    void step_impl(cppSimulatorImp* engine, bool default_action) override ;

    void train(const std::vector<PackedData>& data, int episode) override ;

private:
    int get_default(torch::Tensor x);
    
};