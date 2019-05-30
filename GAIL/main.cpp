#include "Env.h"
#include "util.h"

#include <chrono>


class my_barrier
{

public:
    my_barrier(int count)
        : thread_count(count)
        , counter(0)
        , waiting(0)
    {}

    void wait()
    {
        //fence mechanism
        std::unique_lock<std::mutex> lk(m);
        ++counter;
        ++waiting;
        if (counter == thread_count) {
            lk.unlock();
            cv.notify_all();
        }
        else {
            cv.wait(lk);
        }
        --waiting;
        if (waiting == 0)
        {
            //reset barrier
            counter = 0;
        }
    }

private:
    std::mutex m;
    std::condition_variable cv;
    int counter;
    int waiting;
    int thread_count;
};

const int thread_num = std::thread::hardware_concurrency();
//const int thread_num = 3;

my_barrier barrier1(thread_num + 1);
my_barrier barrier2(thread_num + 1);

std::condition_variable cv1;
std::condition_variable cv2;

std::mutex m1;
std::mutex m2;

class ReplayQueue
{
public:
    ReplayQueue() {}

    void push_back(PackedData data) {
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

ReplayQueue Q;


int main(int argc, char** argv) {
    auto shared_actor_net = std::make_shared<Actor>(10, 9, 128);
    auto shared_critic_net = std::make_shared<Critic>(10, 9, 128);
    auto shared_d_net = std::make_shared<DisCriminator>(10, 9, 128);

    int count = 0;
    if (argc >= 2) {
        count = std::atoi(argv[1]);
        load_model(shared_actor_net, "actor", count);
        load_model(shared_critic_net, "critic", count);
        load_model(shared_d_net, "discriminator", count);
    }

    Env* p_test = new Env(shared_actor_net, shared_critic_net, shared_d_net);

    if (argc == 3 && std::string(argv[2]) == "test") {
        std::cout << p_test->record() << std::endl;
        return 0;
    }

    float lr = 1E-3;

    torch::optim::SGD actor_optim(shared_actor_net->parameters(),
        torch::optim::SGDOptions(lr));

    torch::optim::SGD critic_optim(shared_critic_net->parameters(),
        torch::optim::SGDOptions(lr));

    torch::optim::SGD d_optim(shared_d_net->parameters(),
        torch::optim::SGDOptions(lr));


    std::vector<Env*> vec_env;

    std::vector<std::thread> vec_threads;

    bool first = true;
    for (int i = 0; i < thread_num; ++i) {
        Env* penv = new Env(shared_actor_net, shared_critic_net, shared_d_net);
        vec_env.push_back(penv);
        auto thread_fn = [&](Env* env, bool first) {
            while (true) {
                env->reset();
                while (env->step(first, false));

                PackedData data = env->prepare_data();
                Q.push_back(data);
            }
        };

        vec_threads.emplace_back(thread_fn, penv, first);
        first = false;
    }

    bool expert = true;

    shared_actor_net->to(torch::kCUDA);
    shared_critic_net->to(torch::kCUDA);
    shared_d_net->to(torch::kCUDA);


    while (true) {

        std::vector<PackedData> datas = Q.get_all();
        
        if (datas.empty())
        {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }

        if (datas.size() > std::thread::hardware_concurrency()) {
            datas.resize(std::thread::hardware_concurrency());
        }
        
        for (auto& data : datas)
        {
            count++;
            d_optim.zero_grad();

            data.state = data.state.to(torch::kCUDA);
            data.expert_action = data.expert_action.to(torch::kCUDA);
            data.actor_one_hot = data.actor_one_hot.to(torch::kCUDA);
            data.reward = data.reward.to(torch::kCUDA);

            torch::Tensor expert_prob = shared_d_net->forward(data.state, data.expert_action);
            torch::Tensor expert_label = torch::ones_like(expert_prob);
            torch::Tensor expert_d_loss = torch::binary_cross_entropy(expert_prob, expert_label).mean();

            torch::Tensor actor_action_prob = torch::softmax(shared_actor_net->forward(data.state), 1);
            torch::Tensor actor_prob = shared_d_net->forward(data.state, (actor_action_prob * data.actor_one_hot).detach());
            torch::Tensor actor_label = torch::zeros_like(actor_prob);
            torch::Tensor actor_d_loss = torch::binary_cross_entropy(actor_prob, actor_label).mean();

            torch::Tensor prob_diff = torch::relu(expert_prob - actor_prob).detach();

            torch::Tensor total_d_loss = expert_d_loss + actor_d_loss;

            total_d_loss.backward();

            d_optim.step();

            actor_optim.zero_grad();
            critic_optim.zero_grad();

            torch::Tensor actor_prob2 = shared_d_net->forward(data.state, actor_action_prob * data.expert_action);
            torch::Tensor actor_label2 = torch::ones_like(actor_prob2);

            torch::Tensor actor_d_loss2 = (torch::relu(torch::binary_cross_entropy(actor_prob2, actor_label2) - 0.1))*prob_diff.mean();

            torch::Tensor values = shared_critic_net->forward(data.state);
            torch::Tensor critic_loss = data.reward - values;
            critic_loss = critic_loss * critic_loss;
            critic_loss = critic_loss.mean();

            torch::Tensor adv = data.reward - values.detach();

            torch::Tensor actor_log_probs = torch::log(torch::sum(actor_action_prob * data.actor_one_hot, 1));
            torch::Tensor actor_loss = -actor_log_probs * adv;
            actor_loss = actor_loss.mean();

            torch::Tensor total_loss = actor_d_loss2 + actor_loss + critic_loss;
            total_loss.backward();

            std::stringstream ss;
            ss << "training loss " << toNumber<float>(total_loss) << std::endl;

            critic_optim.step();
            actor_optim.step();
            if (count % 100 == 0) {
                p_test->evaluate();
                save_model(shared_actor_net, "actor", count);
                save_model(shared_critic_net, "critic", count);
                save_model(shared_d_net, "discriminator", count);
            }
        }
       
    }

    return 0;
}
