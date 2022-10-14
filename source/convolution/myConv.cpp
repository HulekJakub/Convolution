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
        std::cout << "dupa1" << std::endl;

        vector<Tensor> results;
        results.reserve(data.size());

        for (auto &&batch : data)
        {
            results.push_back(logic_.runConvolution(batch, weights_, args_.strides(), args_.padding()));
        }

        return ConvData(results);
    }


    void MyConv::setWeights(const vector<Tensor>& weights)
    {
        weights_ = weights;
    }

    void MyConv::setWeights(int channels)
    {
        weights_.reserve(args_.nKernels());
        for (size_t h = 0; h < weights_.capacity(); h++)
        {
            weights_.push_back(Tensor(channels, args_.kernelSize().x(), args_.kernelSize().y()));
        }
    }

    vector<Tensor> MyConv::weights() const
    {
        return weights_;
    }

}
