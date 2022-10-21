#include "convData.hpp"

namespace data
{
    ConvData::ConvDataIterator ConvData::begin() const 
    { 
        return ConvDataIterator(0, this); 
    }
    
    ConvData::ConvDataIterator ConvData::end() const 
    { 
        return ConvDataIterator(data_.size(), this); 
    }

    ConvData::ConvData(int batches, Vec3<int> batch_dims, const vector<float>& data) 
    {
        data_.reserve(batches);
        int batch_size = batch_dims.mul();
        for (size_t i = 0; i < batches; i++)
        {
            auto begin = data.begin() + i * batch_size;
            auto end = data.begin() + (i + 1) * batch_size;
            data_.push_back(Tensor(vector<float>(begin, end), batch_dims));
        }
    }

    std::size_t ConvData::size() const{
        return data_.size();
    }

    Vec3<int> ConvData::batchSize() const
    {
        return data_.front().shape();
    }

    vector<float> ConvData::copyData() const
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

    bool ConvData::empty() const
    {
        return data_.empty();
    }

}