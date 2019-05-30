#pragma once

#include "Models.h"

class MovingMode : public ModelNode {
public:
    explicit MovingMode(int input_dim=10, int hidden_dim=32, int output_dim = 9) :
        ModelNode(input_dim, hidden_dim, output_dim) {}

    const std::string& type() override {
        static std::string type_name = "MovingMode";
        return type_name;
    }

    void step_impl(cppSimulatorImp* engine, bool default_action) override ;
    void train(const std::vector<PackedData>& data, int episode) override ;

};