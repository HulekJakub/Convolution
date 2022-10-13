#include "tensor.hpp"
#include <iostream>
#include "../utils/vec3.hpp"

using std::cout;
using std::endl;

namespace data
{

    Tensor::Tensor(int depth, int width, int height)
    {
        Tensor::init(depth, width, height);
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

    void Tensor::init(int depth, int width, int height)
    {
        if(depth <= 0 || width <= 0 || height <= 0)
        {
            throw new std::invalid_argument("All of the tensor dimensions must be positive");
        }

        data_ = vector<vector<vector<float>>>();
        data_.resize(depth);
        for (auto &&vector1 : data_)
        {
            vector1.resize(width);
            for(auto &&vector2 : vector1)
            {
                vector2.resize(height);
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
                cout << "    ";
                for(auto &&x : vector2)
                {
                    cout << x << ", ";
                }
                cout << endl;
            }
            cout << "  ]," << endl;
        }
        cout << "]" << endl;
    }
}