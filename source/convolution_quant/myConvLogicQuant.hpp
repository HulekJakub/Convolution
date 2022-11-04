#pragma once

#include <vector>
#include "../utils/tensor.hpp"
#include "../data/convArgs.hpp"
#include "../data/convData.hpp"
#include "../utils/vec2.hpp"

using utils::Tensor;
using utils::Vec2;
using std::vector;

namespace convolution_quant
{
    class MyConvLogicQuant
    {
    private:
        Vec3<int> calculateOutputShape(const Vec3<int>& padded_image_shape, int n_kernels, const Vec2<int>& kernel_size, const Vec2<int>& strides) const;
        Tensor<uint8_t> padImage(const Tensor<uint8_t>& image, const vector<int>& padding) const;

        vector<int32_t> calculateForFilter(const Tensor<uint8_t>& image, const Tensor<int8_t>& filter, int32_t bias, const Vec2<int>& strides, const Vec3<int>& layer_output_shape) const;
        int32_t dotSum(const Tensor<uint8_t>& image, const Vec2<int>& start, const Vec2<int>& end, const Tensor<int8_t>& filter) const;
        int32_t activationFunction(int32_t x) const; // ReLU

    public:
        Tensor<int32_t> runConvolution(const Tensor<uint8_t>& image, const vector<Tensor<int8_t>>& weights, vector<int32_t> biases, const Vec2<int>& strides, const vector<int>& padding) const;

    };
    
}