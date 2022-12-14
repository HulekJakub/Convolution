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
#include "myConvLogic.hpp"

using data::ConvData;
using utils::Tensor;
using data::ConvArgs;
using std::vector;
using utils::Vec3;

namespace convolution
{
    class MyConv
    {
    private:
        ConvArgs args_;
        MyConvLogic logic_;
        vector<Tensor<float>> weights_; 
        vector<float> biases_; 

        unsigned long long time_taken_ = 0;

        static float getRandomBias();

    public:
        MyConv(ConvArgs args, MyConvLogic logic=MyConvLogic()): args_(args), logic_(logic) {}
        ConvData<float> execute(ConvData<float> data);
        void setWeights(const vector<Tensor<float>>& weights);
        void setBiases(const vector<float>& biases);
        void setBiases();
        ~MyConv(){}

        unsigned long long timeTaken() const;
        const vector<Tensor<float>>& weights() const;
        const vector<float>& biases() const;

    };
    
}
