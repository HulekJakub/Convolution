#include "onednnConv.hpp"

namespace convolution
{
    ConvData<float> OnednnConv::execute(ConvData<float> data)
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
    
        // Source (src), weights, bias, and destination (dst) tensors
        // dimensions.
        memory::dims src_dims = {N, IC, IH, IW};
        memory::dims weights_dims = {OC, IC, KH, KW};
        memory::dims bias_dims = {OC};
        memory::dims dst_dims = {N, OC, OH, OW};
    
        // Strides, padding dimensions.
        memory::dims strides_dims = {SH, SW};
        memory::dims padding_dims_l = {PH_L, PW_L};
        memory::dims padding_dims_r = {PH_R, PW_R};
    
        // Allocate buffers.
        // Initialize src, weights, and dst tensors.
        std::vector<float> src_data = data.copyData();
        std::vector<float> weights_data = weights_.copyData();
        std::vector<float> bias_data = biases_;
        std::vector<float> dst_data(N * OC * OH * OW);
    
        // Create memory objects for tensor data (src, weights, dst). In this
        // example, NCHW layout is assumed for src and dst, and OIHW for weights.
        auto user_src_mem = memory({src_dims, memory::data_type::f32, memory::format_tag::nchw}, engine);
        auto user_weights_mem = memory({weights_dims, memory::data_type::f32, memory::format_tag::oihw}, engine);
        auto user_dst_mem = memory({dst_dims, memory::data_type::f32, memory::format_tag::nchw}, engine);
    
        // Create memory descriptors with format_tag::any for the primitive. This
        // enables the convolution primitive to choose memory layouts for an
        // optimized primitive implementation, and these layouts may differ from the
        // ones provided by the user.
        auto conv_src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::any);
        auto conv_weights_md = memory::desc(weights_dims, memory::data_type::f32, memory::format_tag::any);
        auto conv_dst_md = memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::any);
    
        // Create memory descriptor and memory object for input bias.
        auto user_bias_md = memory::desc(bias_dims, memory::data_type::f32, memory::format_tag::a);
        auto user_bias_mem = memory(user_bias_md, engine);
    
        // Write data to memory object's handle.
        write_to_dnnl_memory(src_data.data(), user_src_mem);
        write_to_dnnl_memory(weights_data.data(), user_weights_mem);
        write_to_dnnl_memory(bias_data.data(), user_bias_mem);
    
        // Create operation descriptor.
        auto conv_desc = convolution_forward::desc(prop_kind::forward_training,
                algorithm::convolution_direct, conv_src_md, conv_weights_md,
                user_bias_md, conv_dst_md, strides_dims, padding_dims_l,
                padding_dims_r);
    
        // Create primitive post-ops (ReLU).
        const float scale = 1.f;
        const float alpha = 0.f;
        const float beta = 0.f;
        post_ops conv_ops;
        conv_ops.append_eltwise(scale, algorithm::eltwise_relu, alpha, beta);
        primitive_attr conv_attr;
        conv_attr.set_post_ops(conv_ops);
    
        auto start_time = __rdtsc();

        // Create primitive descriptor.
        auto conv_pd = convolution_forward::primitive_desc(conv_desc, conv_attr, engine);
    
        // For now, assume that the src, weights, and dst memory layouts generated
        // by the primitive and the ones provided by the user are identical.
        auto conv_src_mem = user_src_mem;
        auto conv_weights_mem = user_weights_mem;
        auto conv_dst_mem = user_dst_mem;
    
        // Reorder the data in case the src and weights memory layouts generated by
        // the primitive and the ones provided by the user are different. In this
        // case, we create additional memory objects with internal buffers that will
        // contain the reordered data. The data in dst will be reordered after the
        // convolution computation has finalized.
        if (conv_pd.src_desc() != user_src_mem.get_desc()) {
            conv_src_mem = memory(conv_pd.src_desc(), engine);
            reorder(user_src_mem, conv_src_mem)
                    .execute(engine_stream, user_src_mem, conv_src_mem);
        }
    
        if (conv_pd.weights_desc() != user_weights_mem.get_desc()) {
            conv_weights_mem = memory(conv_pd.weights_desc(), engine);
            reorder(user_weights_mem, conv_weights_mem)
                    .execute(engine_stream, user_weights_mem, conv_weights_mem);
        }
    
        if (conv_pd.dst_desc() != user_dst_mem.get_desc()) {
            conv_dst_mem = memory(conv_pd.dst_desc(), engine);
        }
    
        // Create the primitive.
        auto conv_prim = convolution_forward(conv_pd);
    
        // Primitive arguments.
        std::unordered_map<int, memory> conv_args;
        conv_args.insert({DNNL_ARG_SRC, conv_src_mem});
        conv_args.insert({DNNL_ARG_WEIGHTS, conv_weights_mem});
        conv_args.insert({DNNL_ARG_BIAS, user_bias_mem});
        conv_args.insert({DNNL_ARG_DST, conv_dst_mem});
    
        // Primitive execution: convolution with ReLU.
        conv_prim.execute(engine_stream, conv_args);

        // Reorder the data in case the dst memory descriptor generated by the
        // primitive and the one provided by the user are different.
        if (conv_pd.dst_desc() != user_dst_mem.get_desc()) {
            reorder(conv_dst_mem, user_dst_mem)
                    .execute(engine_stream, conv_dst_mem, user_dst_mem);
        } else
            user_dst_mem = conv_dst_mem;
    
        // Wait for the computation to finalize.
        engine_stream.wait();

        auto end_time = __rdtsc();
        time_taken_ += end_time - start_time;

        // Read data from memory object's handle.
        read_from_dnnl_memory(dst_data.data(), user_dst_mem);

        return ConvData<float>(N, Vec3<int>(args_.nKernels(), OH, OW), dst_data);
    }

    unsigned long long OnednnConv::timeTaken() const
    {
        return time_taken_;
    }


    void OnednnConv::setWeights(const ConvData<float>& weights)
    {
        weights_ = weights;
    }

    void OnednnConv::setBiases(const vector<float>& biases)
    {
        biases_ = biases;
    }

    inline void OnednnConv::write_to_dnnl_memory(void *handle, dnnl::memory &mem) 
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

    inline void OnednnConv::read_from_dnnl_memory(void *handle, dnnl::memory &mem) 
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
