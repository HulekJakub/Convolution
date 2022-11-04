#include "myConvQuant.hpp"

using std::invalid_argument;

namespace convolution_quant
{
    ConvData<float> MyConvQuant::execute(ConvData<float> data) 
    {
        std::cout << "Running with args: " << std::endl;
        args_.print();

        if(weights_float_.empty())
        {
            throw std::string("Weights are not initialized");
        }
        if(biases_float_.empty())
        {
            throw std::string("Biases are not initialized");
        }

        if(weights_.empty() || biases_.empty())
        {
            Qa_ = quant_logic_.getInputScalingFactor(data);
            Qw_ = quant_logic_.getWeightsScalingFactor(weights_float_);
            Qb_ = quant_logic_.getBiasScalingFactor(Qa_, Qw_);

            weights_ = quant_logic_.quantizeWeights(weights_float_, Qw_);
            biases_ = quant_logic_.quantizeBiases(biases_float_, Qb_);
        }

        vector<Tensor<float>> results;
        results.reserve(data.size());

        auto start_time = __rdtsc();
        for (auto &&batch : data)
        {
            auto quantized_batch = quant_logic_.qunatizeBatch(batch, Qa_);
            auto result = logic_.runConvolution(quantized_batch, weights_, biases_, args_.strides(), args_.padding());
            results.push_back(quant_logic_.dequnatizeBatch(result, Qa_, Qw_));
        }
        auto end_time = __rdtsc();
        time_taken_ += end_time - start_time;

        return ConvData<float>(results);
    }


    void MyConvQuant::setWeights(const vector<Tensor<float>>& weights)
    {
        if(weights.size() != args_.nKernels() )
        {
            throw invalid_argument("Invalid weights size");
        }
        for(int i = 0; i < weights.size() -1; i++)
        {
            if(weights[i].shape() != weights[i+1].shape())
            {
                throw invalid_argument("All filters must have the same shape");
            }
        }
        
        weights_float_ = weights;
    }

    void MyConvQuant::setBiases(const vector<float>& biases)
    {
        if(biases.size() != args_.nKernels())
        {
            throw invalid_argument("Invalid biases size");
        }
        biases_float_ = biases;
    }

    void MyConvQuant::setBiases()
    {
        biases_ = vector<int32_t> (args_.nKernels(), 0);
        std::generate(biases_.begin(), biases_.end(), getRandomBias);
    }

    const vector<Tensor<int8_t>>& MyConvQuant::weights() const
    {
        return weights_;
    }

    const vector<int32_t>& MyConvQuant::biases() const
    {
        return biases_;
    }

    unsigned long long MyConvQuant::timeTaken() const
    {
        return time_taken_;
    }

    float MyConvQuant::getRandomBias()
    {
        static std::default_random_engine e;
        e.seed(std::chrono::system_clock::now().time_since_epoch().count());
        static std::uniform_real_distribution<> dis(-10, 10);
        return dis(e);
    }
}
