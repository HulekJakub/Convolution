#include "myConvLogic.hpp"

namespace convolution
{

    Tensor MyConvLogic::runConvolution(const Tensor& data, const vector<Tensor>& weights, const Vec2<int>& strides, vector<int> padding) const
    {
        vector<vector<vector<float>>> result;
        //result.reserve(weights.size());

        for (auto &filter : weights)
        {
            //auto filterResults = calculateForFilter(data, filter, strides, padding);

            //result.push_back(filterResults);
        }

        return Tensor(result);
    }

    vector<vector<float>> MyConvLogic::calculateForFilter(const Tensor& data, const Tensor& filter, const Vec2<int>& strides, vector<int> padding) const
    {
        vector<vector<float>> result;
        auto image = data.data();
        auto filter_data = filter.data();

        return result;
    }

    float MyConvLogic::dotSum(const vector<vector<vector<float>>>& a, const vector<vector<vector<float>>>& b) const
    {
        if(a.size() != b.size() || a.front().size() != b.front().size() || a.front().front().size() != b.front().front().size())
        {
            throw new std::invalid_argument("Both matrixes must be of equal shape");
        }

        float dot_sum = 0.0;
        for (size_t i = 0; i < a.size(); i++)
        {
            for (size_t j = 0; j < a.front().size(); j++)
            {
                for (size_t k = 0; k < a.front().front().size(); k++)
                {
                    dot_sum += a[i][j][k] * b[i][j][k];
                }
            }
        }

        return dot_sum;
    }
}
