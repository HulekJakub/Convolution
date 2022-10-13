#pragma once

#include "../utils/vec2.hpp"
#include "../utils/vec3.hpp"

using utils::Vec3;
using utils::Vec2;

namespace data
{
    enum Padding
    {
        VALID,
        SAME
    };

    class ConvArgs
    {
    private:
        int n_kernels_;
        Vec2<int> kernel_size_;
        Vec2<int> strides_;
        Padding padding_;

    public:
        ConvArgs(int n_kernels, Vec2<int> kernel_size, Vec2<int> strides=Vec2<int>(1), Padding padding=Padding::VALID): 
            n_kernels_(n_kernels), 
            kernel_size_(kernel_size), 
            strides_(strides), 
            padding_(padding) {}

        void print() const;
        int nKernels() const;
        Vec2<int> kernelSize() const;
        Vec2<int> strides() const;
        Padding padding() const;
        ~ConvArgs(){}
    };
    

    
}
