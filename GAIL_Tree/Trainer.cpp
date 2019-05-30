//
// Created by len on 5/26/19.
//

#include <functional>

#include "Trainer.h"


Trainer::Trainer() :episode(0){
    init();
}

void Trainer::init() {
    root = std::make_shared<ActionChoice>();
    auto curr = root;
    init_recur(curr);
}

void Trainer::init_recur(std::shared_ptr<ModelNode> node) {
    if (!node) {
        return;
    }

    node->cuda();

    //TODO: name each layer
    Q_map[node];
    name2model[node->type()] = node;

    for (auto c: node->children) {
        init_recur(c);
    }
}

void Trainer::start_training() {
    working_thread_ = std::thread(std::bind(&Trainer::training_thread, this));
}

void Trainer::put(const std::string& name, const PackedData& data) {
    const auto& node = name2model[name];
    Q_map[node].push_back(data);
}

void Trainer::training_thread() {
    while (true) {
        episode++;
        for (auto& p : Q_map) {
            auto& node = p.first;
            auto& Q = p.second;

            auto data = Q.get_all();
            if (data.empty()) {
                // sleep for 1sec if there is no data, avoid busy waiting
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
            node->train(data, episode);
        }
        if (episode % 500 == 0) {
            root->save(".", episode);
        }
    }
}

