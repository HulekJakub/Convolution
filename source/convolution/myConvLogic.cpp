#include "myConvLogic.hpp"

namespace convolution
{

    Tensor MyConvLogic::runConvolution(const Tensor& data, const vector<Tensor>& weights, vector<float> biases, const Vec2<int>& strides, const vector<int>& padding) const
    {
        vector<vector<vector<float>>> result;
        result.reserve(weights.size());

        for (size_t i = 0; i < weights.size(); i++)
        {
            auto filterResults = calculateForFilter(data, weights[i], biases[i], strides, padding);

            result.push_back(filterResults);
        }

        return Tensor(result);
    }

    vector<vector<float>> MyConvLogic::calculateForFilter(const Tensor& data, const Tensor& filter, float bias, const Vec2<int>& strides, const vector<int>& padding) const
    {
        auto image = data.data();
        auto filter_data = filter.data();
        auto padded_image = padImage(image, padding);
        auto filter_size = Vec2<int>(filter_data.front().size(), filter_data.front().front().size());

        vector<vector<float>> result;
        result.reserve((padded_image.front().size() - filter_size.x()) / strides.x());

        for (size_t i = 0; i <= padded_image.front().size() - filter_size.x(); i += strides.x())
        {
            vector<float> vec1d;
            vec1d.reserve((padded_image.front().front().size() - filter_size.y()) / strides.y());
            
            for (size_t j = 0; j <= padded_image.front().front().size() - filter_size.y(); j += strides.y())
            {
                auto sum = dotSum(padded_image, Vec2<int>(i, j), Vec2<int>(i + filter_size.x(), j + filter_size.y()), filter_data);
                sum += bias;
                auto activated_sum = activationFunction(sum);
                vec1d.push_back(activated_sum);
            }
            result.push_back(vec1d);
        }

        return result;
    }

    vector<vector<vector<float>>> MyConvLogic::padImage(const vector<vector<vector<float>>>& image, const vector<int>& padding) const
    {
        vector<vector<vector<float>>> padded_image;
        padded_image.resize(image.size());
        auto new_height = image.front().size() + padding[1] + padding[3];
        auto new_width = image.front().front().size() + padding[0] + padding[2];

        for(size_t h = 0; h < padded_image.size(); h++)
        {
            auto &vec2d = padded_image[h];
            vec2d.reserve(new_height);
            for (size_t i = 0; i < padding[1]; i++)
            {   
                vec2d.push_back(vector<float> (new_width, 0));
            }

            for (size_t i = 0; i < image.front().size(); i++)
            {
                auto vec1d = vector<float> (new_width, 0);
                for (size_t j = padding[0]; j < new_width - padding[2]; j++)
                {
                    vec1d[j] = image[h][i][j - padding[0]];
                }
                vec2d.push_back(vec1d);
            }
            
            for (size_t i = 0; i < padding[3]; i++)
            {   
                vec2d.push_back(vector<float> (new_width, 0));
            }
        }

        return padded_image;
    }

    float MyConvLogic::activationFunction(float x) const
    {
        return x < 0.0 ? 0.0 : x;;
    }

    float MyConvLogic::dotSum(const vector<vector<vector<float>>>& image, const Vec2<int>& start, const Vec2<int>& end, const vector<vector<vector<float>>>& filter) const
    {
        if(image.size() != filter.size() || 
            end.x() - start.x() > image.front().size() || 
            end.y() - start.y() > image.front().front().size() || 
            end.x() - start.x() != filter.front().size() || 
            end.y() - start.y() != filter.front().front().size())
        {
            throw new std::invalid_argument("Both matrixes must be of equal shape");
        }

        float dot_sum = 0.0;
        for (size_t i = 0; i < image.size(); i++)
        {
            for (size_t j = 0; j < end.x() - start.x(); j++)
            {
                for (size_t k = 0; k < end.y() - start.y(); k++)
                {
                    dot_sum += image[i][j + start.x()][k + start.y()] * filter[i][j][k];
                }
            }
        }

        return dot_sum;
    }
}
