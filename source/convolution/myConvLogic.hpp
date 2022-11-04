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
        Vec3<int> calculateOutputShape(const Vec3<int>& padded_image_shape, int n_kernels, const Vec2<int>& kernel_size, const Vec2<int>& strides) const;
        Tensor<float> padImage(const Tensor<float>& image, const vector<int>& padding) const;

        vector<float> calculateForFilter(const Tensor<float>& image, const Tensor<float>& filter, float bias, const Vec2<int>& strides, const Vec3<int>& layer_output_shape) const;
        float dotSum(const Tensor<float>& image, const Vec2<int>& start, const Vec2<int>& end, const Tensor<float>& filter) const;
        float activationFunction(float x) const; // ReLU

    public:
    Tensor<float> runConvolution(const Tensor<float>& data, const vector<Tensor<float>>& weights, vector<float> biases, const Vec2<int>& strides, const vector<int>& padding) const;
    };
    
}