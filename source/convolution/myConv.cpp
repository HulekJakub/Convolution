#include "myConv.hpp"

namespace convolution
{
    ConvData MyConv::run(ConvData data) 
    {
        std::cout << "Running with args: " << std::endl;
        args_.print();

        if(weights_.empty())
        {
            setWeights((*data.begin()).shape().x());
        }

        vector<Tensor> results;
        results.reserve(data.size());

        for (auto &&batch : data)
        {
            results.push_back(logic_.runConvolution(batch, weights_, biases_, args_.strides(), args_.padding()));
        }

        return ConvData(results);
    }


    void MyConv::setWeights(const vector<Tensor>& weights)
    {
        weights_ = weights;
        initBiases();
    }

    void MyConv::setWeights(int channels)
    {
        weights_.reserve(args_.nKernels());
        for (size_t h = 0; h < weights_.capacity(); h++)
        {
            weights_.push_back(Tensor(channels, args_.kernelSize().x(), args_.kernelSize().y()));
        }
        initBiases();
    }

    void MyConv::initBiases()
    {
        biases_ = vector<float> (weights_.size(), 0);
        std::generate(biases_.begin(), biases_.end(), getRandomBias);
    }

    const vector<Tensor>& MyConv::weights() const
    {
        return weights_;
    }

    const vector<float>& MyConv::biases() const
    {
        return biases_;
    }


    float MyConv::getRandomBias()
    {
        static std::default_random_engine e;
        static std::uniform_real_distribution<> dis(-1, 1);
        return dis(e);
    }
}
