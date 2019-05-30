#pragma once

#include "SimDota2/include/SimDota2.h"

#include "ActionChoice.h"
#include "Trainer.h"

class Env {
public:
    Env(Config* cfg, Trainer* trainer);

    bool step();

    void reset();

    void working_thread();

    std::shared_ptr<ModelNode> root;

    Config* config;
    cppSimulatorImp* engine;
private:
    void push_data(const std::shared_ptr<ModelNode>& node);
    void update_param(const std::shared_ptr<ModelNode>& node, const std::shared_ptr<ModelNode>& master_node);

    int tick;
    Trainer* trainer;
    float prev_exp;
    float prev_hp;
};
