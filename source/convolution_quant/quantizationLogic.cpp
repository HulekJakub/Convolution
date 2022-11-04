#include "quantizationLogic.hpp"


namespace convolution_quant
{


    vector<float> QuantizationLogic::getInputScalingFactor(data::ConvData<float> data) const
    {
        auto n_channels = data.batchSize().x();
        auto size_channel = data.batchSize().y() * data.batchSize().z();

        vector<float> result(n_channels, 0);

        for (auto &&image : data)
        {
            auto data_ptr = image.getDataPtr(0);
            int channel = -1;

            for (size_t i = 0; i < image.size(); i++)
            {
                if(i % size_channel == 0){
                    channel++;
                }
                result[channel] = std::max(result[channel], std::abs(*data_ptr));
                ++data_ptr;
            }
        }

        for (size_t i = 0; i < result.size(); i++)
        {
            result[i] = 255.f / result[i];
        }

        return result;
    }

    vector<float> QuantizationLogic::getWeightsScalingFactor(vector<Tensor<float>> weights) const
    {
        auto n_channels = weights.front().shape().x();
        auto size_channel = weights.front().shape().y() * weights.front().shape().z();

        vector<float> result(n_channels, 0);

        for (auto &&filter : weights)
        {
            auto data_ptr = filter.getDataPtr(0);
            int channel = -1;

            for (size_t i = 0; i < filter.size(); i++)
            {
                if(i % size_channel == 0){
                    channel++;
                }
                result[channel] = std::max(result[channel], std::abs(*data_ptr));
                ++data_ptr;
            }
        }

        for (size_t i = 0; i < result.size(); i++)
        {
            result[i] = 127.f / result[i];
        }
        
        return result;
    }

    float QuantizationLogic::getBiasScalingFactor(vector<float> Qa, vector<float> Qw) const
    {
        auto Qa_avg = std::accumulate(Qa.begin(), Qa.end(), 0.0f) / Qa.size();
        auto Qw_avg = std::accumulate(Qw.begin(), Qw.end(), 0.0f) / Qw.size();

        return Qa_avg * Qw_avg;
    }

    vector<Tensor<int8_t>> QuantizationLogic::quantizeWeights(vector<Tensor<float>> weights, vector<float> Qw) const
    {
        auto n_channels = weights.front().shape().x();
        auto size_channel = weights.front().shape().y() * weights.front().shape().z();

        vector<Tensor<int8_t>> result;
        result.reserve(weights.size());

        for (auto &&filter : weights)
        {
            vector<int8_t> new_filter(filter.size(), 0);
            auto data_ptr = filter.getDataPtr(0);
            int channel = -1;

            for (size_t i = 0; i < filter.size(); i++)
            {
                if(i % size_channel == 0){
                    channel++;
                }
                const float scaled = *data_ptr * Qw[channel];
                new_filter[i] = saturate<int8_t>(scaled, -127, 127);
                ++data_ptr;
            }
            result.push_back(Tensor<int8_t>(new_filter, filter.shape()));
        }

        return result;
    }

    vector<int32_t> QuantizationLogic::quantizeBiases(vector<float> biases, float Qb) const
    {
        vector<int32_t> result(biases.size(), 0);

        for (size_t i = 0; i < biases.size(); i++)
        {
            const float scaled = biases[i] * Qb;
            result[i] = saturate<int32_t>(scaled, INT32_MIN, INT32_MAX);
        }

        return result;
    }

    Tensor<uint8_t> QuantizationLogic::qunatizeBatch(Tensor<float> data, vector<float> Qa) const
    {
        auto n_channels = data.shape().x();
        auto size_channel = data.shape().y() * data.shape().z();

        vector<uint8_t> new_data(data.size(), 0);
        auto data_ptr = data.getDataPtr(0);
        int channel = -1;
        float Qac = 0;

        for (size_t i = 0; i < data.size(); i++)
        {
            if(i % size_channel == 0){
                channel++;
                Qac = Qa[channel];
            }
            const float scaled = *data_ptr * Qac;
            new_data[i] = saturate<uint8_t>(scaled, 0, 255);
            ++data_ptr;
        }

        return Tensor<uint8_t>(new_data, data.shape());
    }

    Tensor<float> QuantizationLogic::dequnatizeBatch(Tensor<int32_t> data, vector<float> Qa, vector<float> Qw) const
    {
        auto n_channels = data.shape().x();
        auto size_channel = data.shape().y() * data.shape().z();

        vector<float> new_data(data.size(), 0);
        auto data_ptr = data.getDataPtr(0);
        int channel = -1;
        float Qwa = 0.f;

        for (size_t i = 0; i < data.size(); i++)
        {
            if(i % size_channel == 0){
                channel++;
                Qwa = Qa[channel] * Qw[channel];
            }
            new_data[i] = *data_ptr / Qwa;
            ++data_ptr;
        }

        return Tensor<float>(new_data, data.shape());
    }

}
