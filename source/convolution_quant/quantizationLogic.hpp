#pragma once

#include <vector>
#include <numeric>
#include "../data/convData.hpp"
#include "../utils/tensor.hpp"

using std::vector;
using utils::Tensor;

namespace convolution_quant
{


    class QuantizationLogic
    {
    private:
        template<typename T>
        T saturate(int value, int min, int max) const
        {
            return (T)(value < min ? min : (value > max ? max : value));
        }
    public:
        vector<float> getInputScalingFactor(data::ConvData<float> data) const;
        vector<float> getWeightsScalingFactor(vector<Tensor<float>> weights) const;
        float getBiasScalingFactor(vector<float> Qa, vector<float> Qw) const;

        vector<Tensor<int8_t>> quantizeWeights(vector<Tensor<float>> weights, vector<float> Qw) const;
        vector<int32_t> quantizeBiases(vector<float> biases, float Qb) const;
        Tensor<uint8_t> qunatizeBatch(Tensor<float> data, vector<float> Qa) const;

        Tensor<float> dequnatizeBatch(Tensor<int32_t> data, vector<float> Qa, vector<float> Qw) const;
    };
    
}