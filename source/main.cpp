#include <iostream>
#include <vector>

#include "utils/tensor.hpp"
#include "data/convArgs.hpp"
#include "data/convData.hpp"
#include "utils/vec2.hpp"
#include "utils/vec3.hpp"
#include "convolution/myConv.hpp"
#include "convolution/onednnConv.hpp"

using utils::Tensor;
using std::vector;
using utils::Vec2;
using utils::Vec3;
using data::ConvArgs;
using data::ConvData;
using convolution::MyConv;
using convolution::OnednnConv;

vector<Tensor> generateData(int batches, int channels, int height, int width)
{
  vector<Tensor> res;
  res.reserve(batches);
  for (size_t h = 0; h < batches; h++)
  {
      res.push_back(Tensor(Vec3<int>(channels, height, width)));
  }

  return res;
}

int main(int argc, char** argv)
{
  int n = 1, c = 2, h=5, w=5;
  
  ConvArgs args(4, Vec2<int>(3), Vec2<int>(2), vector<int> {2,2,0,0});
  ConvData data(generateData(n, c, h, w));
  MyConv conv(args);
  conv.setWeights(c);
  conv.setBiases();

  std::cout<<"Data: \n";
  for (auto &&batch : data)
  {
    batch.print();
  }

  try
  {
    auto res = conv.execute(data);
    std::cout<<"Result: \n";
    for (auto &&batch : res)
    {
      batch.print();
    }
  }
  catch(const std::invalid_argument& e)
  {
    std::cout << e.what() << '\n';
  }
  catch(std::string& e)
  {
    std::cout << e << '\n';
  }
 
  OnednnConv a;



}

