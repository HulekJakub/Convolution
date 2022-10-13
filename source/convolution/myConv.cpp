#include "myConv.hpp"

namespace convolution
{
    ConvData MyConv::run(ConvData data) 
    {
        for (auto &&batch : data)
        {
            batch.print();
        }
    }

}
