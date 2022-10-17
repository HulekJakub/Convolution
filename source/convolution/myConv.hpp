#pragma once

#include<iostream>
#include <algorithm>
#include <random>
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
        vector<Tensor> weights_; 
        vector<float> biases_; 

        void initBiases();
        static float getRandomBias();

    public:
        MyConv(ConvArgs args, MyConvLogic logic=MyConvLogic()): args_(args), logic_(logic) {}
        ConvData run(ConvData data);
        void setWeights(const vector<Tensor>& weights);
        void setWeights(int channels);
        ~MyConv(){}

        const vector<Tensor>& weights() const;
        const vector<float>& biases() const;

    };
    
}
