#pragma once

// rdtsc
#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#include <dnnl.hpp>
#include <exception>
#include <stdexcept>

#include "../data/convArgs.hpp"
#include "../data/convData.hpp"
#include "../utils/tensor.hpp"
#include "quantizationLogic.hpp"

using namespace dnnl;
using data::ConvArgs;
using data::ConvData;
using utils::Tensor;

namespace convolution_quant
{
    class OnednnConvQuant
    {
    private:
        ConvArgs args_;
        QuantizationLogic quant_logic_;
        vector<Tensor<float>> weights_; 
        vector<float> biases_; 
        convolution_forward conv;

        vector<float> Qa_; 
        vector<float> Qw_; 
        float Qb_; 

        unsigned long long time_taken_ = 0;

        static float getRandomBias();
        inline void write_to_dnnl_memory(void *handle, dnnl::memory &mem);
        inline void read_from_dnnl_memory(void *handle, dnnl::memory &mem);

    public:
        OnednnConvQuant(ConvArgs args, QuantizationLogic quant_logic=QuantizationLogic()): args_(args), quant_logic_(quant_logic) {}
        ~OnednnConvQuant(){}

        ConvData<float> execute(ConvData<float> data);
        void setWeights(const vector<Tensor<float>>& weights);
        void setBiases(const vector<float>& biases);

        const vector<Tensor<float>>& weights() const;
        const vector<float>& biases() const;

        unsigned long long timeTaken() const;
    };


}
