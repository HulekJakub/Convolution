#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

#include "data/tensor.hpp"
#include "data/convArgs.hpp"
#include "data/convData.hpp"
#include "utils/vec2.hpp"
#include "convolution/myConv.hpp"

using data::Tensor;
using std::vector;
using utils::Vec2;
using data::ConvArgs;
using data::ConvData;
using convolution::MyConv;

float get_random()
{
    static std::default_random_engine e;
    static std::uniform_real_distribution<> dis(0, 1);
    return dis(e);
}

vector<Tensor> generateData(int batches, int channels, int width, int height)
{
  vector<Tensor> res;
  res.reserve(batches);
  for (size_t h = 0; h < batches; h++)
  {
    vector<vector<vector<float>>> data(channels);
    for (auto &&vector2d : data)
    {
      vector2d.resize(width);
      for(auto &&vector1d : vector2d)
      {
        vector2d.reserve(width);
        std::generate(vector1d.begin(), vector1d.end(), get_random);
      }
    }

    res.push_back(Tensor(std::move(data)));
  }
  

  return res;
}

int main(int argc, char** argv)
{
  srand (static_cast <unsigned> (time(0)));

  std::cout << "Started" << std::endl;
  int n = 64, c = 20, h=24, w=24;

  // dnnl::memory::dims strides = {2,2};
  // dnnl::memory::dims padding = {0,0};
  // dnnl::memory::dims ksize = {2,2};
  vector<vector<vector<float>>> vec {{{ 1, 2, 3 }}};

  
  ConvArgs args(256, Vec2<int>(3));
  args.print();

  Tensor a(std::move(vec));
  a.print();

  ConvData data(generateData(n, c, w, h));
  MyConv conv(args);

  conv.run(data);
}

