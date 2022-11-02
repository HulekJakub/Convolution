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
    template <class T>
    class ConvData
    {
    private:
        vector<Tensor<T>> data_;

    public:
        ConvData(vector<Tensor<T>> data): data_(data) {}
        ConvData(int batches, Vec3<int> batch_dims, const vector<float>& data)
        {
            data_.reserve(batches);
            int batch_size = batch_dims.mul();
            for (size_t i = 0; i < batches; i++)
            {
                auto begin = data.begin() + i * batch_size;
                auto end = data.begin() + (i + 1) * batch_size;
                data_.push_back(Tensor<float>(vector<float>(begin, end), batch_dims));
            }
        }
        ConvData(){}
        ~ConvData(){}

        template <class U>
        class ConvDataIterator
        {
        private:
            int idx_;
            const ConvData* data_;
        public:
            ConvDataIterator(int idx, const ConvData<U>* data): idx_(idx), data_(data){}
            ConvDataIterator<U> operator++() { ++idx_; return *this; }
            bool operator!=(const ConvDataIterator<U> & other) const { return this->data_ == other.data_ && idx_ != other.idx_; }
            const Tensor<U>& operator*() const { return data_->data_[idx_]; }

        };

        ConvDataIterator<T> begin() const
        { 
            return ConvDataIterator<T>(0, this); 
        }
        ConvDataIterator<T> end() const
        { 
            return ConvDataIterator<T>(data_.size(), this); 
        }

        bool empty() const
        {
            return data_.empty();
        }
        std::size_t size() const
        {
            return data_.size();
        }
        Vec3<int> batchSize() const
        {
            return data_.front().shape();
        }
        vector<T> copyData() const
        {
            vector<float> result;
            result.reserve(size() * batchSize().mul());
            for(auto &tensor : data_)
            {
                auto copy = tensor.copyData();
                result.insert(result.begin(), copy.begin(), copy.end());
            }
            return result;
        }
    };
}
