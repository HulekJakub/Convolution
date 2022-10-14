#pragma once

#include "../data/convData.hpp"

using data::ConvData;

namespace convolution
{
    class IConv
    {

    public:
        virtual ConvData run(ConvData data) = 0;
    };
}