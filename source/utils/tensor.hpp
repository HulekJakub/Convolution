#pragma once

#include <vector>
#include <memory>
#include "vec3.hpp"

using std::vector;
using std::unique_ptr;

namespace utils
{
    class Tensor
    {
    private:
        vector<vector<vector<float>>> data_;

        void init(int channels, int width, int height);
        static float get_random();
    public:
        Tensor(int channels, int width, int height);
        Tensor(int width, int height);
        Tensor(int height);
        Tensor(vector<vector<vector<float>>> &data);
        const vector<vector<vector<float>>>& data() const;
        utils::Vec3<std::size_t> shape() const;
        void print() const;
        ~Tensor();
    };
}
