#include "spdlog/sinks/basic_file_sink.h"

#include "Env.h"
#include "Trainer.h"

int main(int argc, char** argv) {
    auto loss_logger = spdlog::basic_logger_mt("loss_logger", "loss.log", true);
    auto action_logger = spdlog::basic_logger_mt("action_logger", "action.log", true);

    auto config = ConfigCacheMgr<Config>::getInstance().get("/home/len/PytorchCppCodes/SimDota2/config/Config.json");

    Trainer trainer;
    trainer.start_training();

    int num_thread = std::thread::hardware_concurrency();
    std::vector<Env> envs;
    std::vector<std::thread> threads;
    for (int i = 0; i < num_thread; ++i) {
        envs.emplace_back(config, &trainer);
    }

    for (int i = 0; i < num_thread; ++i) {
        threads.emplace_back(std::bind(&Env::working_thread, &envs[i]));
    }

    for(auto& t:threads){
        t.join();
    }

    trainer.join();
}