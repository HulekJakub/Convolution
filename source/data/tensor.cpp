#include "tensor.hpp"
#include <algorithm>
#include <random>
#include <iostream>
#include "../utils/vec3.hpp"

using std::cout;
using std::endl;

namespace data
{

    Tensor::Tensor(int channels, int width, int height)
    {
        Tensor::init(channels, width, height);
    }

    Tensor::Tensor(int width, int height)
    {
        Tensor::init(1, width, height);
    }

    Tensor::Tensor(int height)
    {
        Tensor::init(1, 1, height);
    }

    Tensor::Tensor(vector<vector<vector<float>>> &&data)
    {
        data_ = data;
    }

    vector<vector<vector<float>>>& Tensor::data()
    {
        return data_;
    }
    
    Tensor::~Tensor()
    {
    }

    void Tensor::init(int channels, int width, int height)
    {
        if(channels <= 0 || width <= 0 || height <= 0)
        {
            throw new std::invalid_argument("All of the tensor dimensions must be positive");
        }

        data_ = vector<vector<vector<float>>>(channels);
        for (auto &&vector2d : data_)
        {
            vector2d.resize(width);
            for(auto &&vector1d : vector2d)
            {
                vector1d.resize(height);
                std::generate(vector1d.begin(), vector1d.end(), get_random);
            }
        }
    }

    utils::Vec3<std::size_t> Tensor::shape() const
    {
        return utils::Vec3<std::size_t>(data_.size(), data_.front().size() ,data_.front().front().size());
    }

    void Tensor::print() const
    {
        auto tensor_shape = shape();

        cout << "Tensor of dimensions (" << tensor_shape.x() << ", " << tensor_shape.y() << ", " << tensor_shape.z() << ")" << endl;
        cout << "[" << endl;
        for (auto &&vector1 : data_)
        {
            cout << "  [" << endl;
            for(auto &&vector2 : vector1)
            {
                cout << "    [";
                for(auto &&x : vector2)
                {
                    cout << x << ", ";
                }
                cout << "]" << endl;
            }
            cout << "  ]," << endl;
        }
        cout << "]" << endl;
    }

    float Tensor::get_random()
    {
        static std::default_random_engine e;
        static std::uniform_real_distribution<> dis(0, 1);
        return dis(e);
    }
}