#pragma once

#include "../utils/tensor.hpp"
#include "../data/convArgs.hpp"
#include "../utils/vec2.hpp"

using utils::Tensor;
using data::Padding;
using utils::Vec2;

namespace convolution
{
    class MyConvLogic
    {

    public:
        Tensor pad(const Tensor& data, Padding padding) const;
        Tensor convolute(const Tensor& data, const vector<Tensor>& weights, const Vec2<int>& strides) const;
    };
    
}
