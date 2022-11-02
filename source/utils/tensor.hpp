#pragma once

#include <vector>
#include <chrono>
#include "vec3.hpp"
#include <algorithm>
#include <random>
#include <iostream>

using std::vector;
using std::cout;
using std::endl;

namespace utils
{
    #ifndef PLS_WORK
    #define PLS_WORK
    


    #endif

    template <class T>
    class Tensor
    {
    private:
        vector<T> data_;
        Vec3<int> shape_;

        static float getRandom()
        {   
            static std::default_random_engine e;
            e.seed(std::chrono::system_clock::now().time_since_epoch().count());
            static std::uniform_real_distribution<> dis(-10, 10);
            return dis(e);
        }

    public:
        Tensor(const vector<T> &data, Vec3<int> shape)
        {
            if(data.size() != shape.mul())
            {
                throw new std::invalid_argument("Data size is not compatible with shape");
            }
            data_ = data;
            shape_ = shape;
        }   

        T get(int c_idx, int h_idx, int w_idx) const
        {
            if (c_idx >= shape_.x() || h_idx >= shape_.y() || w_idx >= shape_.z() )
            {
                throw new std::invalid_argument("Index out of bounds");
            }
            return data_[c_idx * shape_.y() * shape_.z() + h_idx * shape_.z() + w_idx];
        }
        const T* getDataPtr(int idx) const
        {
            return data_.data() + idx;
        }
        vector<T> copyData() const
        {
            return data_;
        }

        const Vec3<int>& shape() const
        {
            return shape_;
        }
        
        void print() const
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

        int size() const
        {
            return shape_.mul();
        }

        ~Tensor(){}

        
        static vector<float> generate_data(Vec3<int> shape)
        {
            auto data = vector<float>(shape.mul());
            std::generate(data.begin(), data.end(), getRandom);
            return data;
        }
    };

}




