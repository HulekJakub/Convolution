#pragma once

// rdtsc
#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#include <dnnl.hpp>
#include <exception>
#include "../data/convArgs.hpp"
#include "../data/convData.hpp"
#include "../utils/tensor.hpp"

using namespace dnnl;
using data::ConvArgs;
using data::ConvData;
using utils::Tensor;

namespace convolution
{
    class OnednnConv
    {
    private:
        ConvArgs args_;
        ConvData<float> weights_; 
        vector<float> biases_; 
        convolution_forward conv;

        unsigned long long time_taken_ = 0;

        static float getRandomBias();
        inline void write_to_dnnl_memory(void *handle, dnnl::memory &mem);
        inline void read_from_dnnl_memory(void *handle, dnnl::memory &mem);

    public:
        OnednnConv(ConvArgs args): args_(args) {}
        ~OnednnConv(){}

        ConvData<float> execute(ConvData<float> data);
        void setWeights(const ConvData<float>& weights);
        void setBiases(const vector<float>& biases);

        const vector<Tensor<float>>& weights() const;
        const vector<float>& biases() const;

        unsigned long long timeTaken() const;
    };


}
