#pragma once

#include "../data/convData.hpp"
#include "iconv.hpp"
#include "../data/convArgs.hpp"

using data::ConvData;
using data::ConvArgs;

namespace convolution
{
    class MyConv : public IConv
    {
    private:
        ConvArgs args_;

        
    public:
        MyConv(ConvArgs args): args_(args) {}
        ConvData run(ConvData data) override;
        ~MyConv(){}
    };
    
}
