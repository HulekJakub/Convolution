#include "myConvLogic.hpp"

namespace convolution
{

    Tensor MyConvLogic::pad(const Tensor& data, Padding padding) const
    {
        return data;
    }

    Tensor MyConvLogic::convolute(const Tensor& data, const vector<Tensor>& kernels, const Vec2<int>& strides) const
    {
        return data;
    }
}
