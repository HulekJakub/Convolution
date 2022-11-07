#include <iostream>
#include <vector>

#include "utils/tensor.hpp"
#include "data/convArgs.hpp"
#include "data/convData.hpp"
#include "utils/vec2.hpp"
#include "utils/vec3.hpp"
#include "convolution/myConv.hpp"
#include "convolution/onednnConv.hpp"
#include "convolution_quant/myConvQuant.hpp"
#include "convolution_quant/onednnConvQuant.hpp"

using utils::Tensor;
using std::vector;
using std::cout;
using std::endl;
using utils::Vec2;
using utils::Vec3;
using data::ConvArgs;
using data::ConvData;
using convolution::MyConv;
using convolution::OnednnConv;
using convolution_quant::MyConvQuant;
using convolution_quant::OnednnConvQuant;

vector<Tensor<float>> generateData(int batches, int channels, int height, int width)
{
  vector<Tensor<float>> res;
  res.reserve(batches);

  for (size_t h = 0; h < batches; h++)
  {
      auto dims = Vec3<int>(channels, height, width);
      auto random_data = Tensor<float>::generate_data(dims);
      res.push_back(Tensor<float>(random_data, dims));
  }

  return res;
}

vector<Tensor<float>> generateOnes(int batches, int channels, int height, int width)
{
  vector<Tensor<float>> res;
  res.reserve(batches);

  for (size_t h = 0; h < batches; h++)
  {
    vector<float> data(channels * height * width, 1.f);
    res.push_back(Tensor<float>(data, Vec3<int>(channels, height, width)));
  }

  return res;
}

int main(int argc, char** argv)
{
  // Data and args setup
  int n = 8, c = 3, h=256, w=256;
  int n_kernels = 8, kernel_size = 3;
  auto padding = vector<int> {2,2,2,2};
  auto stride = 1;
  
  ConvArgs args(n_kernels, Vec2<int>(kernel_size), Vec2<int>(stride), padding);
  ConvData<float> data(generateData(n, c, h, w));

  std::cout<<"Data: \n";
  for (auto &&batch : data)
  {
    // batch.print();
  }

//=====================My=================================
  MyConv myConv(args);
  myConv.setWeights(generateOnes(n_kernels, c, kernel_size, kernel_size));
  myConv.setBiases();

  auto iters = 100;

  for (size_t i = 0; i < iters; i++)
  {
    try
    {
      cout << "Myconv iter: " << i << endl;
      auto res = myConv.execute(data);
      cout<<"Result: \n";
      for (auto &&batch : res)
      {
        // batch.print();
      }
    }
    catch(const std::invalid_argument& e)
    {
      cout << e.what() << endl;
    }
    catch(std::string& e)
    {
      cout << e << endl;
    }
  }
  
//=====================Onednn=================================

 
  std::cout << std::string(100, '-') << std::endl;
  // OneDnn convolution
  OnednnConv onednnConv(args);
  onednnConv.setWeights(ConvData<float>(myConv.weights()));
  onednnConv.setBiases(myConv.biases());

  for (size_t i = 0; i < iters; i++)
  {
    try
    {
      cout << "Onednn iter: " << i << endl;
      auto res = onednnConv.execute(data);
      cout<<"Result: \n";
      for (auto &&batch : res)
      {
        // batch.print();
      }
    }
    catch(const std::invalid_argument& e)
    {
      cout << e.what() << endl;
    }
    catch(std::string& e)
    {
      cout << e << endl;
    }
  }

//=====================My quantized=================================

  MyConvQuant myConvQuant(args);
  myConvQuant.setWeights(myConv.weights());
  myConvQuant.setBiases(myConv.biases());

  for (size_t i = 0; i < iters; i++)
  {
    try
    {
      cout << "Onednn iter: " << i << endl;
      auto res = myConvQuant.execute(data);
      cout<<"Result: \n";
      for (auto &&batch : res)
      {
        // batch.print();
      }
    }
    catch(const std::invalid_argument& e)
    {
      cout << e.what() << endl;
    }
    catch(std::string& e)
    {
      cout << e << endl;
    }
  }

  //=====================Onednn Quantized=================================

  std::cout << std::string(100, '-') << std::endl;
  OnednnConvQuant onednnConvQuant(args);
  onednnConvQuant.setWeights(myConv.weights());
  onednnConvQuant.setBiases(myConv.biases());

  for (size_t i = 0; i < iters; i++)
  {
    try
    {
      cout << "Onednn iter: " << i << endl;
      auto res = onednnConvQuant.execute(data);
      cout<<"Result: \n";
      for (auto &&batch : res)
      {
        // batch.print();
      }
    }
    catch(const std::invalid_argument& e)
    {
      cout << e.what() << endl;
    }
    catch(std::string& e)
    {
      cout << e << endl;
    }
  }


  cout << "My conv average time: " << myConv.timeTaken() / iters << endl;
  cout << "Onednn conv average time: " << onednnConv.timeTaken() / iters << endl;
  cout << "My conv quant average time: " << myConvQuant.timeTaken() / iters << endl;
  cout << "Onednn convquant average time: " << onednnConvQuant.timeTaken() / iters << endl;

  auto max = std::max({myConv.timeTaken(), onednnConv.timeTaken(), myConvQuant.timeTaken(), onednnConvQuant.timeTaken()});

  cout << "My conv average time of max: " << myConv.timeTaken() / (1.0  * max) << endl;
  cout << "Onednn conv average time of max: " << onednnConv.timeTaken() / (1.0  * max) << endl;
  cout << "My conv quant average time of max: " << myConvQuant.timeTaken() / (1.0  * max) << endl;
  cout << "Onednn convquant average time of max: " << onednnConvQuant.timeTaken() / (1.0  * max) << endl;

  // for:
  // int n = 1, c = 2, h=5, w=5;
  // int n_kernels = 1, kernel_size = 3;
  // auto padding = vector<int> {2,2,2,2}
  // auto stride = 2;
  // My conv average time: 13651
  // Onednn conv average time: 6208
  // my was 2 times slower

  // for:
  // int n = 5, c = 3, h=256, w=256;
  // int n_kernels = 32, kernel_size = 5;
  // auto padding = vector<int> {0,0,0,0};
  // auto stride = 2;
  // My conv average time: 18897972098
  // Onednn conv average time: 7846451
  // my was 2400 times slower
}

