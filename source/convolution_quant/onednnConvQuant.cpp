#include "onednnConvQuant.hpp"

namespace convolution_quant
{
    ConvData<float> OnednnConvQuant::execute(ConvData<float> data)
    {
        if(weights_.empty())
        {
            throw std::string("Weights are not initialized");
        }
        if(biases_.empty())
        {
            throw std::string("Biases are not initialized");
        }

        std::cout << "Running with args: " << std::endl;
        args_.print();

        // Create execution dnnl::engine.
        auto engine = dnnl::engine(dnnl::engine::kind::cpu,0);
 
        // Create dnnl::stream.
        dnnl::stream engine_stream(engine);
    
        auto batch_size = data.batchSize(); 
        // Tensor dimensions.
        const memory::dim N = data.size(), // batch size
                IC = batch_size.x(), // input channels
                IH = batch_size.y(), // input height
                IW = batch_size.z(), // input width
                OC = args_.nKernels(), // output channels
                KH = args_.kernelSize().x(), // weights height
                KW = args_.kernelSize().y(), // weights width
                PH_L = args_.padding()[1], // height padding: left
                PH_R = args_.padding()[3], // height padding: right
                PW_L = args_.padding()[0], // width padding: left
                PW_R = args_.padding()[2], // width padding: right
                SH = args_.strides().x(), // height-wise stride
                SW = args_.strides().y(), // width-wise stride
                OH = (IH - KH + PH_L + PH_R) / SH + 1, // output height
                OW = (IW - KW + PW_L + PW_R) / SW + 1; // output width
    


        //[Configure tensor shapes]
        memory::dims src_dims = {N, IC, IH, IW};
        memory::dims weights_dims = {OC, IC, KH, KW};
        memory::dims bias_dims = {OC};
        memory::dims dst_dims = {N, OC, OH, OW};
        memory::dims strides_dims = {SH, SW};
        memory::dims padding_dims_l = {PH_L, PW_L};
        memory::dims padding_dims_r = {PH_R, PW_R};
        //[Configure tensor shapes]

        //[Choose scaling factors]
        // Choose scaling factors for input, weight, output and bias quantization
        // Choose channel-wise scaling factors for convolution
        Qa_ = quant_logic_.getInputScalingFactor(data);
        Qw_ = quant_logic_.getWeightsScalingFactor(weights_);
        Qb_ = quant_logic_.getBiasScalingFactor(Qa_, Qw_);

        std::vector<float> src_scales = Qa_;
        std::vector<float> weight_scales = Qw_;
        std::vector<float> bias_scales = {Qb_};
        std::vector<float> dst_scales(OC, 1.f);

        std::vector<float> conv_scales(OC, 1.f);
        float scale_out = 1.f;
        int size_channel = OH * OW;

        int channel = -1;
        for (size_t i = 0; i < OC; i++)
        {
            if(i % size_channel == 0){
                channel++;
            }
            dst_scales[i] = scale_out / (Qa_[channel] * Qw_[channel]);
        }

        //[Set scaling mask]
        const int src_mask = 0;
        const int weight_mask = 0;
        const int bias_mask = 0;
        const int dst_mask = 0;
        const int conv_mask = 2; // 1 << output_channel_dim

        // Allocate input and output buffers for user data
        std::vector<float> user_src = data.copyData();
        std::vector<float> user_dst(N * OC * OH * OW);

        // Allocate and fill buffers for weights and bias
        std::vector<float> conv_weights = ConvData<float>(weights_).copyData();
        std::vector<float> conv_bias = biases_;

        //[Allocate buffers]
        auto user_src_memory = memory({{src_dims}, memory::data_type::f32, memory::format_tag::nchw}, engine);
        write_to_dnnl_memory(user_src.data(), user_src_memory);
        auto user_weights_memory
                = memory({{weights_dims}, memory::data_type::f32, memory::format_tag::oihw}, engine);
        write_to_dnnl_memory(conv_weights.data(), user_weights_memory);
        auto user_bias_memory = memory({{bias_dims}, memory::data_type::f32, memory::format_tag::x}, engine);
        write_to_dnnl_memory(conv_bias.data(), user_bias_memory);

        //[Create convolution memory descriptors]
        auto conv_src_md = memory::desc({src_dims}, memory::data_type::u8, memory::format_tag::any);
        auto conv_bias_md = memory::desc({bias_dims}, memory::data_type::s32, memory::format_tag::any);
        auto conv_weights_md = memory::desc({weights_dims}, memory::data_type::s8, memory::format_tag::any);
        auto conv_dst_md = memory::desc({dst_dims}, memory::data_type::f32, memory::format_tag::any);

        //[Create convolution descriptor]
        auto conv_desc = convolution_forward::desc(prop_kind::forward,
                algorithm::convolution_direct, conv_src_md, conv_weights_md,
                conv_bias_md, conv_dst_md, strides_dims, padding_dims_l,
                padding_dims_r);

        //[Configure scaling]
        primitive_attr conv_attr;
        conv_attr.set_output_scales(conv_mask, conv_scales);

        //[Configure post-ops]
        const float ops_scale = 1.f;
        const float ops_alpha = 0.f; // relu negative slope
        const float ops_beta = 0.f;
        post_ops ops;
        ops.append_eltwise(ops_scale, algorithm::eltwise_relu, ops_alpha, ops_beta);
        conv_attr.set_post_ops(ops);

        // check if int8 convolution is supported
        try {
            convolution_forward::primitive_desc(conv_desc, conv_attr, engine);
        } catch (error &e) {
            if (e.status == dnnl_unimplemented)
                throw "oneDNN does not have int8 convolution implementation that supports this system.\nPlease refer to the developer guide for details.";
                        
            // on any other error just re-throw
            throw;
        }

        //[Create convolution primitive descriptor]
        auto conv_prim_desc = convolution_forward::primitive_desc(conv_desc, conv_attr, engine);

        //[Quantize weights]
        auto conv_weights_memory = memory(conv_prim_desc.weights_desc(), engine);
        primitive_attr weight_attr;
        weight_attr.set_output_scales(weight_mask, weight_scales);
        auto weight_reorder_pd
                = reorder::primitive_desc(engine, user_weights_memory.get_desc(), engine,
                        conv_weights_memory.get_desc(), weight_attr);
        auto weight_reorder = reorder(weight_reorder_pd);
        weight_reorder.execute(engine_stream, user_weights_memory, conv_weights_memory);

        auto conv_bias_memory = memory(conv_prim_desc.bias_desc(), engine);
        primitive_attr bias_attr;
        bias_attr.set_output_scales(bias_mask, bias_scales);
        auto bias_reorder_pd
                = reorder::primitive_desc(engine, user_bias_memory.get_desc(), engine,
                        conv_bias_memory.get_desc(), bias_attr);
        auto bias_reorder = reorder(bias_reorder_pd);
        bias_reorder.execute(engine_stream, user_bias_memory, conv_bias_memory);

        auto start_time = __rdtsc();
       
        //[Quantize data]
        auto conv_src_memory = memory(conv_prim_desc.src_desc(), engine);
        primitive_attr src_attr;
        src_attr.set_output_scales(src_mask, src_scales);
        auto src_reorder_pd
                = reorder::primitive_desc(engine, user_src_memory.get_desc(), engine,
                        conv_src_memory.get_desc(), src_attr);
        auto src_reorder = reorder(src_reorder_pd);
        src_reorder.execute(engine_stream, user_src_memory, conv_src_memory);

        auto conv_dst_memory = memory(conv_prim_desc.dst_desc(), engine);

        //[Create convolution primitive]
        auto conv = convolution_forward(conv_prim_desc);
        conv.execute(engine_stream,
                {{DNNL_ARG_SRC, conv_src_memory},
                        {DNNL_ARG_WEIGHTS, conv_weights_memory},
                        {DNNL_ARG_BIAS, conv_bias_memory},
                        {DNNL_ARG_DST, conv_dst_memory}});

        //[Create convolution primitive]
        auto user_dst_memory = memory({{dst_dims}, memory::data_type::f32, memory::format_tag::nchw}, engine);
        write_to_dnnl_memory(user_dst.data(), user_dst_memory);
        primitive_attr dst_attr;
        dst_attr.set_output_scales(dst_mask, dst_scales);
        auto dst_reorder_pd
                = reorder::primitive_desc(engine, conv_dst_memory.get_desc(), engine,
                        user_dst_memory.get_desc(), dst_attr);
        auto dst_reorder = reorder(dst_reorder_pd);
        dst_reorder.execute(engine_stream, conv_dst_memory, user_dst_memory);

        //[Dequantize the result]
        engine_stream.wait();

        auto end_time = __rdtsc();
        time_taken_ += end_time - start_time;

        // Read data from memory object's handle.
        vector<float> dequantized_data(N * OC * OH * OW);
        read_from_dnnl_memory(dequantized_data.data(), user_dst_memory);

        return ConvData<float>(N, Vec3<int>(args_.nKernels(), OH, OW), dequantized_data);
    }

    unsigned long long OnednnConvQuant::timeTaken() const
    {
        return time_taken_;
    }

    void OnednnConvQuant::setWeights(const vector<Tensor<float>>& weights)
    {
        weights_ = weights;
    }

    void OnednnConvQuant::setBiases(const vector<float>& biases)
    {
        biases_ = biases;
    }

    inline void OnednnConvQuant::write_to_dnnl_memory(void *handle, dnnl::memory &mem) 
    {
        dnnl::engine eng = mem.get_engine();
        size_t size = mem.get_desc().get_size();

        if (!handle) throw std::runtime_error("handle is nullptr.");

        if (eng.get_kind() == dnnl::engine::kind::cpu) {
            uint8_t *dst = static_cast<uint8_t *>(mem.get_data_handle());
            if (!dst) throw std::runtime_error("get_data_handle returned nullptr.");
            for (size_t i = 0; i < size; ++i)\
            {
                dst[i] = ((uint8_t *)handle)[i];
            }
        }
    }

    inline void OnednnConvQuant::read_from_dnnl_memory(void *handle, dnnl::memory &mem) 
    {
        dnnl::engine eng = mem.get_engine();
        size_t size = mem.get_desc().get_size();

        if (!handle) throw std::runtime_error("handle is nullptr.");

        if (eng.get_kind() == dnnl::engine::kind::cpu) {
            uint8_t *src = static_cast<uint8_t *>(mem.get_data_handle());
            if (!src) throw std::runtime_error("get_data_handle returned nullptr.");
            for (size_t i = 0; i < size; ++i)
            {
                ((uint8_t *)handle)[i] = src[i];
            }
        }
    }
}
