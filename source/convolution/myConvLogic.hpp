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
        float activationFunction(float x) const; // ReLU
        vector<vector<float>> calculateForFilter(const Tensor& data, const Tensor& filter, const Vec2<int>& strides, const vector<int>& padding) const;
        float dotSum(const vector<vector<vector<float>>>& a, const Vec2<int>& start, const Vec2<int>& end, const vector<vector<vector<float>>>& b) const;
        vector<vector<vector<float>>> padImage(const vector<vector<vector<float>>>& image, const vector<int>& padding) const;


    public:
        Tensor runConvolution(const Tensor& data, const vector<Tensor>& weights, const Vec2<int>& strides, const vector<int>& padding) const;
    };
    
}