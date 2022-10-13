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

    std::size_t ConvData::size() const{
        return data_.size();
    }
}