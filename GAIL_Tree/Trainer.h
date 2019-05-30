//
// Created by len on 5/26/19.
//

#pragma once

#include <map>
#include <atomic>

#include "ActionChoice.h"

class ReplayQueue
{
public:
    ReplayQueue() = default;

    void push_back(const PackedData& data) {
        std::lock_guard<std::mutex> g(mtx);
        q_.push_back(data);
    }

    std::vector<PackedData> get_all() {
        std::lock_guard<std::mutex> g(mtx);
        std::vector<PackedData> empty_q;
        q_.swap(empty_q);
        return empty_q;
    }

private:
    std::mutex mtx;
    std::vector<PackedData> q_;
};



class Trainer {
public:
    Trainer();

    void init();
    void init_recur(std::shared_ptr<ModelNode> node);

    void start_training();
    void put(const std::string& name, const PackedData& data);
    void join() { working_thread_.join(); }
    std::shared_ptr<ModelNode> get_root() { return root; }

private:
    void training_thread();

    int episode;

    std::thread working_thread_;

    std::shared_ptr<ModelNode> root;
    std::unordered_map<std::string, std::shared_ptr<ModelNode>> name2model;
    std::unordered_map<std::shared_ptr<ModelNode>, ReplayQueue> Q_map;
};
