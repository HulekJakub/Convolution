#include <iostream>
#include "convArgs.hpp"
#include "../utils/vec2.hpp"

using std::cout;
using std::endl;

namespace data
{
    int ConvArgs::nKernels() const { return n_kernels_; }
    Vec2<int> ConvArgs::kernelSize() const { return kernel_size_; }
    Vec2<int> ConvArgs::strides() const { return strides_; }
    Padding ConvArgs::padding() const { return padding_; }

    void ConvArgs::print() const
    {
        cout << "Kernels: " << n_kernels_ << endl; 
        cout << "Kernel size: "; kernel_size_.print();
        cout << "Strides: "; strides_.print();
        cout << "Padding: " << (padding_ == Padding::SAME ? "Same" : "Valid") << endl;
    }
}
