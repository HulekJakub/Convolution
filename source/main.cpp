#include <iostream>
#include <vector>

#include "utils/tensor.hpp"
#include "data/convArgs.hpp"
#include "data/convData.hpp"
#include "utils/vec2.hpp"
#include "convolution/myConv.hpp"

using utils::Tensor;
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
  int n = 1, c = 2, h=5, w=5;
  
  ConvArgs args(1, Vec2<int>(3), Vec2<int>(1), vector<int> {1,1,0,0});
  ConvData data(generateData(n, c, w, h));
  MyConv conv(args);

  auto res = conv.run(data);

  std::cout<<"Data: \n";
  for (auto &&batch : data)
  {
    batch.print();
  }
  std::cout<<"Result: \n";
  for (auto &&batch : res)
  {
    batch.print();
  }

}

