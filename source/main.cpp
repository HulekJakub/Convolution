#include <iostream>
#include <vector>

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


vector<Tensor> generateData(int batches, int channels, int width, int height)
{
  vector<Tensor> res;
  res.reserve(batches);
  for (size_t h = 0; h < batches; h++)
  {
    res.push_back(Tensor(channels, width, height));
  }
  
  return res;
}

int main(int argc, char** argv)
{
  int n = 2, c = 3, h=4, w=4;
  
  ConvArgs args(256, Vec2<int>(3));
  ConvData data(generateData(n, c, w, h));
  MyConv conv(args);

  auto res = conv.run(data);

  for (auto &&batch : res)
  {
    batch.print();
  }
}

