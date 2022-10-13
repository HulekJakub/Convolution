#include <iostream>
#include "convArgs.hpp"
#include "../utils/vec2.hpp"

using std::cout;
using std::endl;

namespace data
{
    void ConvArgs::print() const
    {
        cout << "Kernels: " << n_kernels_ << endl; 
        cout << "Kernel size: "; kernel_size_.print();
        cout << "Strides: "; strides_.print();
        cout << "Padding: " << (padding_ == Padding::SAME ? "Same" : "Valid") << endl;
    }
}
