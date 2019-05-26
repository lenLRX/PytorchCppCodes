#include "Actor.h"
#include "Critic.h"
#include "DisCriminator.h"

#include "SimDota2/include/SimDota2.h"

typedef std::map<std::string, torch::Tensor> PackedData;

class ModelNode {
public:
    ModelNode(int input_dim, int hidden_dim, int output_dim) :
        input_dim_(input_dim), hidden_dim_(hidden_dim), output_dim_(output_dim) {}

    virtual const std::string& type() = 0;

    void load(const std::string& path, int i);

    void save(const std::string& path, int i);

    virtual void step(cppSimulatorImp* engine, bool default_action) = 0;

    void set_reward(float reward);

    void reset_node();

    void update_param(std::shared_ptr<Actor> actor_master,
        std::shared_ptr<Critic> critic_master,
        std::shared_ptr<DisCriminator> d_master);

    virtual PackedData training_data();

    virtual void train(const std::vector<PackedData>& data) = 0;

    std::vector<std::shared_ptr<ModelNode> > children;

protected:

    int input_dim_;
    int hidden_dim_;
    int output_dim_;

    std::shared_ptr<Actor> actor_;
    std::shared_ptr<Critic> critic_;
    std::shared_ptr<DisCriminator> discriminator_;

    std::vector<torch::Tensor> states;
    std::vector<torch::Tensor> rewards;
    std::vector<torch::Tensor> expert_actions;
    std::vector<torch::Tensor> actor_actions;
private:
    void load_impl(const std::string& path, std::shared_ptr<torch::nn::Module> m, const std::string& prefix, int i);
    void save_impl(const std::string& path, std::shared_ptr<torch::nn::Module> m, const std::string& prefix, int i);
};