#pragma once
#include <dnnl.hpp>
#include "../data/convArgs.hpp"

using namespace dnnl;

namespace convolution
{
    class OnednnConv
    {
    private:
        convolution_forward conv;

        void init();
    public:
        OnednnConv(/* args */);
        ~OnednnConv();
    };
    
    OnednnConv::OnednnConv(/* args */)
    {
    }
    
    OnednnConv::~OnednnConv()
    {
    }
    

}
