#include "myConvLogic.hpp"

namespace convolution
{

    Tensor MyConvLogic::runConvolution(const Tensor& image, const vector<Tensor>& weights, vector<float> biases, const Vec2<int>& strides, const vector<int>& padding) const
    {
        auto padded_image = padImage(image, padding);
        Vec2<int> kernel_size(weights.front().shape().y(), weights.front().shape().z());
        Vec3<int> output_shape = calculateOutputShape(padded_image.shape(), weights.size(), kernel_size, strides);

        vector<float> result;
        result.reserve(output_shape.mul());

        for (size_t i = 0; i < weights.size(); i++)
        {
            auto filterResults = calculateForFilter(padded_image, weights[i], biases[i], strides, output_shape);
            result.insert(result.end(), filterResults.begin(), filterResults.end());
        }

        return Tensor(result, output_shape);
    }

    Vec3<int> MyConvLogic::calculateOutputShape(const Vec3<int>& padded_image_shape, int n_kernels, const Vec2<int>& kernel_size, const Vec2<int>& strides) const
    {
        int depth = n_kernels;
        int height = 1 + (padded_image_shape.y() - kernel_size.x()) / strides.x(); 
        int width = 1 + (padded_image_shape.z() - kernel_size.y()) / strides.y(); 
        return Vec3<int>(depth, height, width);
    }


    Tensor MyConvLogic::padImage(const Tensor& image, const vector<int>& padding) const
    {
        if(padding == vector<int>(4, 0))
        {
            return image;
        }
        
        vector<float> padded_image_data;
        Vec3<int> padded_image_shape
        (
            image.shape().x(), 
            image.shape().y() + padding[1] + padding[3], 
            image.shape().z()+ padding[0] + padding[2]
        );

        padded_image_data.reserve(padded_image_shape.mul());
        vector<float> row_padding(padded_image_shape.z(), 0);


        for(size_t c = 0; c < padded_image_shape.x(); c++)
        {
            for (size_t i = 0; i < padding[1]; i++)
            {   
                padded_image_data.insert(padded_image_data.end(), row_padding.begin(), row_padding.end());
            }

            for (size_t i = 0; i < image.shape().y(); i++)
            {
                auto vec1d = vector<float> (padded_image_shape.z(), 0);
                for (size_t j = padding[0]; j < padded_image_shape.z() - padding[2]; j++)
                {
                    vec1d[j] = image.get(c, i, j - padding[0]);
                }
                padded_image_data.insert(padded_image_data.end(), vec1d.begin(), vec1d.end());
            }
            
            for (size_t i = 0; i < padding[3]; i++)
            {   
                padded_image_data.insert(padded_image_data.end(), row_padding.begin(), row_padding.end());
            }
        }

        return Tensor(padded_image_data, padded_image_shape);
    }

    vector<float> MyConvLogic::calculateForFilter(const Tensor& image, const Tensor& filter, float bias, const Vec2<int>& strides, const Vec3<int>& layer_output_shape) const
    {
        auto filter_height = filter.shape().y();
        auto filter_width = filter.shape().z();

        vector<float> result;
        result.reserve(layer_output_shape.y() * layer_output_shape.z());

        for (size_t i = 0; i < layer_output_shape.y(); i++)
        {   
            for (size_t j = 0; j < layer_output_shape.z(); j++)
            {
                Vec2<int> start(i * strides.x(), j * strides.y());
                Vec2<int> end(i * strides.x() + filter_height, j * strides.y() + filter_width);
                auto sum = dotSum(image, start, end, filter);
                sum += bias;
                auto activated_sum = activationFunction(sum);
                result.push_back(activated_sum);
            }
        }

        return result;
    }

    float MyConvLogic::activationFunction(float x) const
    {
        return x < 0.0 ? 0.0 : x;
    }

    float MyConvLogic::dotSum(const Tensor& image, const Vec2<int>& start, const Vec2<int>& end, const Tensor& filter) const
    {
        float dot_sum = 0.f;
        for (size_t i = 0; i < image.shape().x(); i++)
        {
            for (size_t j = 0; j < end.x() - start.x(); j++)
            {
                for (size_t k = 0; k < end.y() - start.y(); k++)
                {
                    dot_sum += image.get(i, j + start.x(), k + start.y()) * filter.get(i, j, k);
                }
            }
        }
        
        return dot_sum;
    }
}
