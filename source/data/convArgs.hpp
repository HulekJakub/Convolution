#pragma once

#include <vector>
#include "../utils/vec2.hpp"
#include "../utils/vec3.hpp"

using utils::Vec3;
using utils::Vec2;

namespace data
{
    class ConvArgs
    {
    private:
        int n_kernels_;
        Vec2<int> kernel_size_;
        Vec2<int> strides_;
        std::vector<int> padding_;

    public:
        ConvArgs(int n_kernels, Vec2<int> kernel_size, Vec2<int> strides=Vec2<int>(1), std::vector<int> padding=std::vector<int>{0,0,0,0}): 
            n_kernels_(n_kernels), 
            kernel_size_(kernel_size), 
            strides_(strides), 
            padding_(padding) 
        {
            if(padding_.size() != 4)
            {
                throw new std::invalid_argument("Padding must be a vector of size 4");
            }
        }

        void print() const;
        int nKernels() const;
        Vec2<int> kernelSize() const;
        Vec2<int> strides() const;
        std::vector<int> padding() const;
        ~ConvArgs(){}
    };
}
