#pragma once

#include "Models.h"

class LastHitMode : public ModelNode {
public:
    explicit LastHitMode(int input_dim = 3, int hidden_dim = 32, int output_dim = 1) :
        ModelNode(input_dim, hidden_dim, output_dim) {}

    const std::string& type() override {
        static const std::string type_name = "LastHItMode";
        return type_name;
    }

    void train(const std::vector<PackedData>& data, int episode) override ;

    void step_impl(cppSimulatorImp* engine, bool default_action) override ;
};