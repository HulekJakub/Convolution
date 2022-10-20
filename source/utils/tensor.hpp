#pragma once

#include <vector>
#include <memory>
#include <chrono>
#include "vec3.hpp"

using std::vector;
using std::unique_ptr;

namespace utils
{
    class Tensor
    {
    private:
        vector<float> data_;
        Vec3<int> shape_;


        void init(Vec3<int> shape);
        static float getRandom();
    public:
        Tensor(Vec3<int> shape);
        Tensor(const vector<float> &data, Vec3<int> shape);
        float get(int c_idx, int h_idx, int w_idx) const;
        const Vec3<int>& shape() const;
        void print() const;
        int size() const;
        ~Tensor();
    };
}
