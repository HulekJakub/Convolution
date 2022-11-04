#pragma once

// rdtsc
#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#include <iostream>
#include <algorithm>
#include <random>
#include <chrono>
#include "../data/convData.hpp"
#include "../data/convArgs.hpp"
#include "../utils/tensor.hpp"
#include "../utils/vec3.hpp"
#include "myConvLogicQuant.hpp"
#include "quantizationLogic.hpp"

using data::ConvData;
using utils::Tensor;
using data::ConvArgs;
using std::vector;
using utils::Vec3;

namespace convolution_quant
{
    class MyConvQuant
    {
    private:
        ConvArgs args_;
        MyConvLogicQuant logic_;
        QuantizationLogic quant_logic_;
        
        vector<Tensor<int8_t>> weights_; 
        vector<int32_t> biases_; 
        vector<float> Qa_; 
        vector<float> Qw_; 
        float Qb_; 

        vector<Tensor<float>> weights_float_; 
        vector<float> biases_float_; 

        unsigned long long time_taken_ = 0;

        static float getRandomBias();

    public:
        MyConvQuant(ConvArgs args, MyConvLogicQuant logic=MyConvLogicQuant(), QuantizationLogic qunat_logic=QuantizationLogic()): args_(args), logic_(logic), quant_logic_(qunat_logic) {}
        ConvData<float> execute(ConvData<float> data);
        void setWeights(const vector<Tensor<float>>& weights);
        void setBiases(const vector<float>& biases);
        void setBiases();
        ~MyConvQuant(){}

        unsigned long long timeTaken() const;
        const vector<Tensor<int8_t>>& weights() const;
        const vector<int32_t>& biases() const;

    };
    
}
