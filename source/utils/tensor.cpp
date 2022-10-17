#include "tensor.hpp"
#include <algorithm>
#include <random>
#include <iostream>
#include "../utils/vec3.hpp"

using std::cout;
using std::endl;

namespace utils
{

    Tensor::Tensor(int channels, int height, int width)
    {
        Tensor::init(channels, height, width);
    }

    Tensor::Tensor(int height, int width)
    {
        Tensor::init(1, height, width);
    }

    Tensor::Tensor(int width)
    {
        Tensor::init(1, 1, width);
    }

    Tensor::Tensor(vector<vector<vector<float>>> &data)
    {
        data_ = data;
    }

    const vector<vector<vector<float>>>& Tensor::data() const
    {
        return data_;
    }
    
    Tensor::~Tensor()
    {
    }

    void Tensor::init(int channels, int height, int width)
    {
        if(channels <= 0 || height <= 0 || width <= 0)
        {
            throw new std::invalid_argument("All of the tensor dimensions must be positive");
        }

        data_ = vector<vector<vector<float>>>(channels);
        for (auto &&vector2d : data_)
        {
            vector2d.resize(height);
            for(auto &&vector1d : vector2d)
            {
                vector1d.resize(width);
                std::generate(vector1d.begin(), vector1d.end(), get_random);
            }
        }
    }

    utils::Vec3<std::size_t> Tensor::shape() const
    {
        auto x = data_.size();
        auto y = x != 0 ? data_.front().size() : 0;
        auto z = y != 0 ? data_.front().front().size() : 0;
        return utils::Vec3<std::size_t>(x, y, z);
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