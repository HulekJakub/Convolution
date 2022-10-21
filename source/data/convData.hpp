#pragma once

#include <vector>
#include <memory>
#include "../utils/tensor.hpp"
#include "../utils/vec3.hpp"

using utils::Tensor;
using std::vector;
using std::unique_ptr;
using utils::Vec3;

namespace data
{

    class ConvData
    {
    private:
        vector<Tensor> data_;

    public:
        ConvData(vector<Tensor> data): data_(data) {}
        ConvData(int batches, Vec3<int> batch_dims, const vector<float>& data);
        ConvData(){}
        ~ConvData(){}

        class ConvDataIterator 
        {
        private:
            int idx_;
            const ConvData* data_;
        public:
            ConvDataIterator(int idx, const ConvData* data): idx_(idx), data_(data){}
            ConvDataIterator operator++() { ++idx_; return *this; }
            bool operator!=(const ConvDataIterator & other) const { return this->data_ == other.data_ && idx_ != other.idx_; }
            const Tensor& operator*() const { return data_->data_[idx_]; }

        };

        ConvDataIterator begin() const;
        ConvDataIterator end() const;
        bool empty() const;
        std::size_t size() const;
        Vec3<int> batchSize() const;
        vector<float> copyData() const;
    };
    
}