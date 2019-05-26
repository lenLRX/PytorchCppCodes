#include "Env.h"

Env::Env(Config* cfg):
    config(cfg), engine(nullptr){

}

bool Env::step() {
    auto hero = engine->getHero("Radiant", 0);

    double time_ = engine->get_time();
    
    if (time_ > 200) {
        return false;
    }

    auto hero = engine->getHero("Radiant", 0);
    if (hero->isDead()) {
        return false;
    }

    root.step(engine, false);

    ///engine running
    engine->loop();
    ///engine running

    float now_exp = hero->getData().exp;
    float now_hp = hero->get_HP();

    float exp_reward = now_exp > prev_exp ? 1 : 0;
    float hp_reward = now_hp < prev_hp ? -1 : 0;

    root.set_reward(exp_reward + hp_reward);

    prev_exp = now_exp;
    prev_hp = now_hp;

    return true;
}

void Env::reset() {
    if (engine) {
        delete engine;
        engine = nullptr;
    }
    engine = new cppSimulatorImp(config);

    auto hero = engine->getHero("Radiant", 0);

    prev_exp = hero->getData().exp;
    prev_hp = hero->get_HP();

    root.reset_node();
}
