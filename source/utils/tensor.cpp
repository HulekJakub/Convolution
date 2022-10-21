#include "tensor.hpp"
#include <algorithm>
#include <random>
#include <iostream>
#include "../utils/vec3.hpp"

using std::cout;
using std::endl;

namespace utils
{

    Tensor::Tensor(Vec3<int> shape)
    {
        Tensor::init(shape);
    }

    Tensor::Tensor(const vector<float> &data, Vec3<int> shape)
    {
        if(data.size() != shape.mul())
        {
            throw new std::invalid_argument("Data size is not compatible with shape");
        }
        data_ = data;
        shape_ = shape;
    }   

    float Tensor::get(int c_idx, int h_idx, int w_idx) const
    {
        if (c_idx >= shape_.x() || h_idx >= shape_.y() || w_idx >= shape_.z() )
        {
            throw new std::invalid_argument("Index out of bounds");
        }
        return data_[c_idx * shape_.y() * shape_.z() + h_idx * shape_.z() + w_idx];
    }
    
    Tensor::~Tensor()
    {
    }

    void Tensor::init(Vec3<int> shape)
    {
        if(shape.x() <= 0 || shape.y() <= 0 || shape.z() <= 0)
        {
            throw new std::invalid_argument("All of the tensor dimensions must be positive");
        }
        shape_ = shape;

        data_ = vector<float>(shape_.mul());

        std::generate(data_.begin(), data_.end(), getRandom);
    }

    const Vec3<int>& Tensor::shape() const
    {
        return shape_;
    }

    void Tensor::print() const
    {
        cout << "Tensor of dimensions "; shape_.print();
        cout << "[" << endl;
        for (size_t i = 0; i < shape_.x(); i++)
        {
            cout << "  [" << endl;
            for (size_t j = 0; j < shape_.y(); j++)
            {
                cout << "    [";
                for (size_t k = 0; k < shape_.z(); k++)
                {
                    cout << get(i, j, k) << ", ";
                }
                cout << "]," << endl;
            }
            cout << "  ]," << endl; 
        }
        cout << "]" << endl;
    }

    int Tensor::size() const
    {
        return shape_.mul();
    }

    float Tensor::getRandom()
    {
        static std::default_random_engine e;
        e.seed(std::chrono::system_clock::now().time_since_epoch().count());
        static std::uniform_real_distribution<> dis(0, 1);
        return dis(e);
    }
}