#include "Env.h"

Env::Env(Config* cfg, Trainer* trainer):
    tick(0), trainer(trainer), config(cfg), engine(nullptr){
    root = std::make_shared<ActionChoice>();
}

void Env::working_thread() {
    while (true) {
        reset();
        while (step());
        push_data(root);

    }
}

bool Env::step() {
    auto hero = engine->getHero("Radiant", 0);

    double time_ = engine->get_time();
    
    if (time_ > 200) {
        return false;
    }

    if (hero->isDead()) {
        return false;
    }

    prev_exp = hero->getData().exp;
    prev_hp = hero->get_HP();

    root->step(engine, false, tick);

    ///engine running
    engine->loop();
    ///engine running

    float now_exp = hero->getData().exp;
    float now_hp = hero->get_HP();

    float exp_reward = now_exp > prev_exp ? 1 : 0;
    float hp_reward = now_hp < prev_hp ? -1 : 0;

    root->set_reward(exp_reward + hp_reward, tick);

    ++tick;

    return true;
}

void Env::reset() {
    tick = 0;
    if (engine) {
        delete engine;
        engine = nullptr;
    }
    engine = new cppSimulatorImp(config);

    auto hero = engine->getHero("Radiant", 0);

    prev_exp = hero->getData().exp;
    prev_hp = hero->get_HP();

    root->reset_node();
    update_param(root, trainer->get_root());
}

void Env::push_data(const std::shared_ptr<ModelNode>& node) {
    trainer->put(node->type(), node->training_data());
    for (auto& c:node->children) {
        push_data(c);
    }
}

void Env::update_param(const std::shared_ptr<ModelNode>& node, const std::shared_ptr<ModelNode>& master_node){
    if (node->type() != master_node->type()) {
        std::stringstream error_msg;
        error_msg << "different node type: " << node->type() << " vs " << master_node->type();
        throw std::runtime_error(error_msg.str());
    }
    node->update_param(master_node->get_actor(),
            master_node->get_critic(),
            master_node->get_discriminator());


    for (size_t i = 0;i < node->children.size(); ++i) {
        update_param(node->children[i], master_node->children[i]);
    }
}
