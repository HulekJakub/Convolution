#pragma once

#include<iostream>
#include "../data/convData.hpp"
#include "../data/convArgs.hpp"
#include "../utils/tensor.hpp"
#include "../utils/vec3.hpp"
#include "myConvLogic.hpp"
#include "iconv.hpp"

using data::ConvData;
using utils::Tensor;
using data::ConvArgs;
using std::vector;
using utils::Vec3;

namespace convolution
{
    class MyConv : public IConv
    {
    private:
        ConvArgs args_;
        MyConvLogic logic_;
        vector<Tensor> weights_; 

    public:
        MyConv(ConvArgs args, MyConvLogic logic=MyConvLogic()): args_(args), logic_(logic) {}
        ConvData run(ConvData data) override;
        void setWeights(const vector<Tensor>& weights);
        void setWeights(int channels);
        ~MyConv(){}

        vector<Tensor> weights() const;
    };
    
}
