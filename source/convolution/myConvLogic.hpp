#pragma once

#include <vector>
#include "../utils/tensor.hpp"
#include "../data/convArgs.hpp"
#include "../utils/vec2.hpp"

using utils::Tensor;
using utils::Vec2;
using std::vector;

namespace convolution
{
    class MyConvLogic
    {
    private:
        vector<vector<float>> calculateForFilter(const Tensor& data, const Tensor& filter, const Vec2<int>& strides, vector<int> padding) const;
        float dotSum(const vector<vector<vector<float>>>& a, const vector<vector<vector<float>>>& b) const;


    public:
        Tensor runConvolution(const Tensor& data, const vector<Tensor>& weights, const Vec2<int>& strides, vector<int> padding) const;
    };
    
}