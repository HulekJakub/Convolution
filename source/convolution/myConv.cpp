#include "myConv.hpp"

using std::invalid_argument;

namespace convolution
{
    ConvData<float> MyConv::execute(ConvData<float> data) 
    {
        std::cout << "Running with args: " << std::endl;
        args_.print();

        if(weights_.empty())
        {
            throw std::string("Weights are not initialized");
        }
        if(biases_.empty())
        {
            throw std::string("Biases are not initialized");
        }

        vector<Tensor<float>> results;
        results.reserve(data.size());

        auto start_time = __rdtsc();
        for (auto &&batch : data)
        {
            results.push_back(logic_.runConvolution(batch, weights_, biases_, args_.strides(), args_.padding()));
        }
        auto end_time = __rdtsc();
        time_taken_ += end_time - start_time;

        return ConvData<float>(results);
    }


    void MyConv::setWeights(const vector<Tensor<float>>& weights)
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
        
        weights_ = weights;
    }

    void MyConv::setWeights(int channels)
    {
        weights_.reserve(args_.nKernels());
        for (size_t h = 0; h < weights_.capacity(); h++)
        {
            auto dims = Vec3<int>(channels, args_.kernelSize().x(), args_.kernelSize().y());
            auto random_data = Tensor<float>::generate_data(dims);
            weights_.push_back(Tensor<float>(random_data, dims));
        }
    }

    void MyConv::setBiases(const vector<float>& biases)
    {
        if(biases.size() != args_.nKernels())
        {
            throw invalid_argument("Invalid biases size");
        }
        biases_ = biases;
    }

    void MyConv::setBiases()
    {
        biases_ = vector<float> (args_.nKernels(), 0);
        std::generate(biases_.begin(), biases_.end(), getRandomBias);
    }

    const vector<Tensor<float>>& MyConv::weights() const
    {
        return weights_;
    }

    const vector<float>& MyConv::biases() const
    {
        return biases_;
    }

    unsigned long long MyConv::timeTaken() const
    {
        return time_taken_;
    }

    float MyConv::getRandomBias()
    {
        static std::default_random_engine e;
        e.seed(std::chrono::system_clock::now().time_since_epoch().count());
        static std::uniform_real_distribution<> dis(-1, 1);
        return dis(e);
    }
}
