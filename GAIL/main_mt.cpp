#include "Env.h"
#include "util.h"

class Worker
{
public:
    Worker() {}

    void start_training() {
        std::thread _thd([this] { while (1) { train(); }});
        thd = std::move(_thd);
    }

    void train() {
        Env env(actor_model, critic_model, d_model);
        env.reset();
        while (env.step(false, true));
        actor_model->zero_grad();
        critic_model->zero_grad();
        d_model->zero_grad();
        env.train_discriminator(false);
        env.train_critic(false);
        env.train_actor(false);
        ps->apply_gradient(*this);
    }

    void evaluate() {
        Env env(actor_model, critic_model, d_model);
        env.evaluate();
    }

    void set_ps(std::shared_ptr<Worker> ps_) {
        ps = ps_;
    }

    void clone(const Worker& other) {
        actor_model = std::dynamic_pointer_cast<Actor>(other.actor_model->clone());
        critic_model = std::dynamic_pointer_cast<Critic>(other.critic_model->clone());
        d_model = std::dynamic_pointer_cast<DisCriminator>(other.d_model->clone());
    }

    void apply_gradient(const Worker& other) {
        const float lr = 1E-3;
        update_module(actor_model, other.actor_model, lr);
        update_module(critic_model, other.critic_model, lr);
        update_module(d_model, other.d_model, lr);
        evaluate();
    }

    void update_module(std::shared_ptr<torch::nn::Module> mine,
        std::shared_ptr<torch::nn::Module> their, float lr) {
        //std::lock_guard<std::mutex> g(mtx);

        auto& mine_params = mine->named_parameters();
        auto& their_params = their->named_parameters();

        for (auto& kv : their_params) {
            auto it = mine_params.find(kv.key());
            if (it == nullptr) {
                throw std::runtime_error("not exits tensor " + kv.key());
            }
            (*it) = (*it) + lr * kv.value().grad();
            //std::cout << kv.key() << " " << lr * kv.value().grad() << std::endl;
        }
    }

    std::shared_ptr<Actor> actor_model;
    std::shared_ptr<Critic> critic_model;
    std::shared_ptr<DisCriminator> d_model;
    std::shared_ptr<Worker> ps;
    std::mutex mtx;
    std::thread thd;
};

int main(int argc, char** argv) {
    auto actor_model = std::make_shared<Actor>(10, 9, 16);
    auto critic_model = std::make_shared<Critic>(10, 9, 16);
    auto d_model = std::make_shared<DisCriminator>(10, 9, 16);

    int count = 26300;
    load_model(actor_model, "actor", count);
    load_model(critic_model, "discriminator", count);
    load_model(d_model, "critic", count);

    int cpu_num = std::thread::hardware_concurrency();
    std::cout << "concurrent num " << cpu_num << std::endl;

    std::shared_ptr<Worker> ps = std::make_shared<Worker>();
    ps->actor_model = actor_model;
    ps->critic_model = critic_model;
    ps->d_model = d_model;

    std::vector<Worker*> workers;
    //workers.resize(cpu_num);
    for (int i = 0; i < cpu_num; ++i) {
        workers.push_back(new Worker());
    }

    for (auto worker : workers) {
        worker->set_ps(ps);
        worker->clone(*ps);
        worker->start_training();
    }

    for (auto worker : workers) {
        worker->thd.join();
    }
}