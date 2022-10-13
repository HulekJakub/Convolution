#pragma once

#include<iostream>
#include "../data/convData.hpp"
#include "../data/convArgs.hpp"
#include "../data/tensor.hpp"
#include "myConvLogic.hpp"
#include "iconv.hpp"

using data::ConvData;
using data::Tensor;
using data::ConvArgs;
using std::vector;


namespace convolution
{
    class MyConv : public IConv
    {
    private:
        ConvArgs args_;
        MyConvLogic logic_;
        vector<Tensor> kernels; 

        
    public:
        MyConv(ConvArgs args, MyConvLogic logic=MyConvLogic()): args_(args), logic_(logic) {}
        ConvData run(ConvData data) const override;
        ~MyConv(){}
    };
    
}
