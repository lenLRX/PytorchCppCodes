#pragma once

#include <torch/torch.h>

template <typename M>
void save_model(M& model, const std::string& prefix, int i) {
    torch::serialize::OutputArchive model_save;
    model->save(model_save);
    model_save.save_to(prefix + "_"  + std::to_string(i) + ".model");
}


template <typename M>
void load_model(M& model, const std::string& prefix, int i) {
    torch::serialize::InputArchive infile;
    infile.load_from(prefix + "_" + std::to_string(i) + ".model");
    model->load(infile);
}
