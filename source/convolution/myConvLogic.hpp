#pragma once

#include <vector>
#include "../utils/tensor.hpp"
#include "../data/convArgs.hpp"
#include "../utils/vec2.hpp"

using utils::Tensor;
using utils::Vec2;
using std::vector;
using Tens = Tensor<float>;

namespace convolution
{
    class MyConvLogic
    {
    private:
        Vec3<int> calculateOutputShape(const Vec3<int>& padded_image_shape, int n_kernels, const Vec2<int>& kernel_size, const Vec2<int>& strides) const;
        Tens padImage(const Tens& image, const vector<int>& padding) const;

        vector<float> calculateForFilter(const Tens& image, const Tens& filter, float bias, const Vec2<int>& strides, const Vec3<int>& layer_output_shape) const;
        float dotSum(const Tens& image, const Vec2<int>& start, const Vec2<int>& end, const Tens& filter) const;
        float activationFunction(float x) const; // ReLU

    public:
    Tens runConvolution(const Tens& data, const vector<Tens>& weights, vector<float> biases, const Vec2<int>& strides, const vector<int>& padding) const;
    };
    
}