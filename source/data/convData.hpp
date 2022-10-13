#pragma once

#include <vector>
#include <memory>
#include "tensor.hpp"


using std::vector;
using std::unique_ptr;

namespace data
{

    class ConvData
    {
    private:
        vector<Tensor> data_;

    public:
        ConvData(vector<Tensor> data): data_(data) {}
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

        ConvDataIterator begin() const { return ConvDataIterator(0, this); }
        ConvDataIterator end() const { return ConvDataIterator(data_.size(), this); }
    };
    
}