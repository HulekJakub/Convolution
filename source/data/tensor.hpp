#pragma once

#include <vector>
#include <memory>
#include "../utils/vec3.hpp"

using std::vector;
using std::unique_ptr;

namespace data
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
        Tensor(vector<vector<vector<float>>> &&data);
        vector<vector<vector<float>>>& data();
        utils::Vec3<std::size_t> shape() const;
        void print() const;
        ~Tensor();
    };
}
