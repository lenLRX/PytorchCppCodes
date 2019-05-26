#pragma once

#include "SimDota2/include/SimDota2.h"

#include "ActionChoice.h"

class Env {
public:
    Env(Config* cfg);

    bool step();

    void reset();

    ActionChoice root;

    Config* config;
    cppSimulatorImp* engine;
private:
    float prev_exp;
    float prev_hp;
};
