#pragma once

#include "../data/convData.hpp"

using data::ConvData;

namespace convolution
{
    class IConv
    {

    public:
        IConv(){}
        virtual ConvData run(ConvData data);
        virtual ~IConv() { }
    };
}