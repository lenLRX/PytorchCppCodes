#pragma once

#include <torch/torch.h>

template<typename T>
T toNumber(torch::Tensor x) {
    return x.item().to<T>();
}
