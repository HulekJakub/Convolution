#include "myConv.hpp"

namespace convolution
{
    ConvData MyConv::run(ConvData data) const
    {
        std::cout << "Running with args: " << std::endl;
        args_.print();

        vector<Tensor> results;
        results.reserve(data.size());

        for (auto &&batch : data)
        {
            auto padded = logic_.pad(batch, args_.padding());
            results.push_back(logic_.convolute(padded, kernels, args_.strides()));
        }

        return data;
    }

}
