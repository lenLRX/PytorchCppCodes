#include "Models.h"

void ModelNode::reset_node() {
    states.clear();
    expert_actions.clear();
    actor_actions.clear();
    rewards.clear();

    for (auto& c : children) {
        c->reset_node();
    }
}

void ModelNode::load(const std::string& path, int i) {
    load_impl(path, actor_, "actor", i);
    load_impl(path, critic_, "critic", i);
    load_impl(path, discriminator_, "discriminator", i);
    for (auto& c : children) {
        c->load(path, i);
    }
}

void ModelNode::save(const std::string& path, int i) {
    save_impl(path, actor_, "actor", i);
    save_impl(path, critic_, "critic", i);
    save_impl(path, discriminator_, "discriminator", i);
    for (auto& c : children) {
        c->save(path, i);
    }
}

void ModelNode::set_reward(float reward) {
    torch::Tensor r = torch::zeros({ 1 });
    r[0] = reward;
    rewards.push_back(r);

    for (auto& c : children) {
        c->set_reward(reward);
    }
}

void ModelNode::load_impl(const std::string& path,
    std::shared_ptr<torch::nn::Module> m, const std::string& prefix, int i) {
    torch::serialize::InputArchive infile;
    infile.load_from(path + "/" + type() + "_" + prefix + "_" + std::to_string(i) + ".model");
    m->load(infile);
}

void ModelNode::save_impl(const std::string& path, 
    std::shared_ptr<torch::nn::Module> m, const std::string& prefix, int i) {
    torch::serialize::OutputArchive model_save;
    m->save(model_save);
    model_save.save_to(path + "/" + type() + "_" + prefix + "_" + std::to_string(i) + ".model");
}

void ModelNode::update_param(std::shared_ptr<Actor> actor_master,
    std::shared_ptr<Critic> critic_master,
    std::shared_ptr<DisCriminator> d_master) {
    actor_ = std::dynamic_pointer_cast<Actor>(actor_master->clone());
    actor_->eval();
    critic_ = std::dynamic_pointer_cast<Critic>(critic_master->clone());
    critic_->eval();
    discriminator_ = std::dynamic_pointer_cast<DisCriminator>(d_master->clone());
    discriminator_->eval();
}

PackedData ModelNode::training_data() {
    PackedData ret;
    ret["state"] = torch::stack(states);
    ret["expert_action"] = torch::stack(expert_actions);
    ret["actor_action"] = torch::stack(actor_actions);
    ret["reward"] = torch::stack(rewards);
    return ret;
}

