// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// Optimized 2x2 deconvolution 1D implementation with pack4 support

// Pack4 optimized version for 2x2 kernel, stride 2
static int deconv1d_k2s2_pack4_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt)
{
    const int w = bottom_blob.w;
    const int inch = bottom_blob.c;
    const int outw = top_blob.w;
    const int outch = top_blob.c;

    const float* bias = bias_data;
    const int bias_term = bias_data.empty() ? 0 : 1;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        Mat out = top_blob.channel(p);
        
        // Initialize with bias (pack4)
#if __ARM_NEON
        float32x4_t _bias0 = bias_term ? vld1q_f32(bias + p * 4) : vdupq_n_f32(0.f);
        out.fill(_bias0);
#else
        if (bias_term)
        {
            const float* bias_ptr = bias + p * 4;
            float* outptr = out;
            for (int i = 0; i < outw * 4; i += 4)
            {
                outptr[i] = bias_ptr[0];
                outptr[i + 1] = bias_ptr[1]; 
                outptr[i + 2] = bias_ptr[2];
                outptr[i + 3] = bias_ptr[3];
            }
        }
        else
        {
            out.fill(0.f);
        }
#endif

        for (int q = 0; q < inch; q++)
        {
            float* outptr = out;
            const Mat img = bottom_blob.channel(q);
            const float* inptr = img;
            
            // Get kernel weights for this input-output channel pair
            // weight layout: outch/4 - inch/4 - kw - 4o - 4i
            const float* kptr = (const float*)weight_data.channel(p).row(q);
            
#if __ARM_NEON
            // Load kernel weights
            // Each kernel position has 4x4 weight matrix (4 output channels x 4 input channels)
            float32x4_t k00 = vld1q_f32(kptr + 0);   // k[0] for out_ch[0-3] from in_ch[0]
            float32x4_t k01 = vld1q_f32(kptr + 4);   // k[0] for out_ch[0-3] from in_ch[1]
            float32x4_t k02 = vld1q_f32(kptr + 8);   // k[0] for out_ch[0-3] from in_ch[2]
            float32x4_t k03 = vld1q_f32(kptr + 12);  // k[0] for out_ch[0-3] from in_ch[3]
            
            float32x4_t k10 = vld1q_f32(kptr + 16);  // k[1] for out_ch[0-3] from in_ch[0]
            float32x4_t k11 = vld1q_f32(kptr + 20);  // k[1] for out_ch[0-3] from in_ch[1]
            float32x4_t k12 = vld1q_f32(kptr + 24);  // k[1] for out_ch[0-3] from in_ch[2]
            float32x4_t k13 = vld1q_f32(kptr + 28);  // k[1] for out_ch[0-3] from in_ch[3]

            // Process input positions
            for (int j = 0; j < w; j++)
            {
                // Load input vector (4 channels)
                float32x4_t input_vec = vld1q_f32(inptr + j * 4);
                
                // Calculate output positions (stride=2)
                float* out_pos0 = outptr + (j * 2) * 4;     // position j*2
                float* out_pos1 = outptr + (j * 2 + 1) * 4; // position j*2+1

                // Accumulate for position j*2 (kernel position 0)
                float32x4_t sum0 = vld1q_f32(out_pos0);
                sum0 = vfmaq_laneq_f32(sum0, k00, input_vec, 0);
                sum0 = vfmaq_laneq_f32(sum0, k01, input_vec, 1);
                sum0 = vfmaq_laneq_f32(sum0, k02, input_vec, 2);
                sum0 = vfmaq_laneq_f32(sum0, k03, input_vec, 3);
                vst1q_f32(out_pos0, sum0);

                // Accumulate for position j*2+1 (kernel position 1)
                float32x4_t sum1 = vld1q_f32(out_pos1);
                sum1 = vfmaq_laneq_f32(sum1, k10, input_vec, 0);
                sum1 = vfmaq_laneq_f32(sum1, k11, input_vec, 1);
                sum1 = vfmaq_laneq_f32(sum1, k12, input_vec, 2);
                sum1 = vfmaq_laneq_f32(sum1, k13, input_vec, 3);
                vst1q_f32(out_pos1, sum1);
            }
#else
            // Scalar fallback for non-NEON
            for (int j = 0; j < w; j++)
            {
                const float* input_ch = inptr + j * 4;
                
                for (int out_ch = 0; out_ch < 4; out_ch++)
                {
                    for (int in_ch = 0; in_ch < 4; in_ch++)
                    {
                        float val = input_ch[in_ch];
                        
                        // Position j*2 (kernel position 0)
                        float* out0 = outptr + (j * 2) * 4 + out_ch;
                        *out0 += val * kptr[in_ch * 4 + out_ch];
                        
                        // Position j*2+1 (kernel position 1)
                        float* out1 = outptr + (j * 2 + 1) * 4 + out_ch;
                        *out1 += val * kptr[16 + in_ch * 4 + out_ch];
                    }
                }
            }
#endif
        }

        // Apply activation
        if (activation_type != 0)
        {
            float* outptr = out;
            int size = outw * 4;
            
#if __ARM_NEON
            for (int i = 0; i < size; i += 4)
            {
                float32x4_t _out = vld1q_f32(outptr + i);
                _out = activation_ps(_out, activation_type, activation_params);
                vst1q_f32(outptr + i, _out);
            }
#else
            for (int i = 0; i < size; i++)
            {
                outptr[i] = activation_ss(outptr[i], activation_type, activation_params);
            }
#endif
        }
    }

    return 0;
}

// Pack1to4 version for input pack1 to output pack4
static int deconv1d_k2s2_pack1to4_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt)
{
    const int w = bottom_blob.w;
    const int inch = bottom_blob.c;
    const int outw = top_blob.w;
    const int outch = top_blob.c;

    const float* bias = bias_data;
    const int bias_term = bias_data.empty() ? 0 : 1;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        Mat out = top_blob.channel(p);
        
        // Initialize with bias (pack4)
#if __ARM_NEON
        float32x4_t _bias0 = bias_term ? vld1q_f32(bias + p * 4) : vdupq_n_f32(0.f);
        out.fill(_bias0);
#else
        if (bias_term)
        {
            const float* bias_ptr = bias + p * 4;
            float* outptr = out;
            for (int i = 0; i < outw * 4; i += 4)
            {
                outptr[i] = bias_ptr[0];
                outptr[i + 1] = bias_ptr[1]; 
                outptr[i + 2] = bias_ptr[2];
                outptr[i + 3] = bias_ptr[3];
            }
        }
        else
        {
            out.fill(0.f);
        }
#endif

        for (int q = 0; q < inch; q++)
        {
            float* outptr = out;
            const float* inptr = bottom_blob.channel(q);
            
            // Get kernel weights for this input-output channel pair
            // weight layout: outch/4 - inch - kw - 4o
            const float* kptr = (const float*)weight_data.channel(p).row(q);
            
#if __ARM_NEON
            // Load kernel weights for 4 output channels
            float32x4_t k0 = vld1q_f32(kptr);      // k[0] for out_ch[0-3]
            float32x4_t k1 = vld1q_f32(kptr + 4);  // k[1] for out_ch[0-3]

            // Process input positions
            for (int j = 0; j < w; j++)
            {
                float input_val = inptr[j];
                float32x4_t input_vec = vdupq_n_f32(input_val);
                
                // Output position j*2 (kernel position 0)
                float* out_pos0 = outptr + (j * 2) * 4;
                float32x4_t sum0 = vld1q_f32(out_pos0);
                sum0 = vfmaq_f32(sum0, k0, input_vec);
                vst1q_f32(out_pos0, sum0);

                // Output position j*2+1 (kernel position 1)
                float* out_pos1 = outptr + (j * 2 + 1) * 4;
                float32x4_t sum1 = vld1q_f32(out_pos1);
                sum1 = vfmaq_f32(sum1, k1, input_vec);
                vst1q_f32(out_pos1, sum1);
            }
#else
            // Scalar fallback
            for (int j = 0; j < w; j++)
            {
                float val = inptr[j];
                
                for (int out_ch = 0; out_ch < 4; out_ch++)
                {
                    // Position j*2 (kernel position 0)
                    float* out0 = outptr + (j * 2) * 4 + out_ch;
                    *out0 += val * kptr[out_ch];
                    
                    // Position j*2+1 (kernel position 1)
                    float* out1 = outptr + (j * 2 + 1) * 4 + out_ch;
                    *out1 += val * kptr[4 + out_ch];
                }
            }
#endif
        }

        // Apply activation
        if (activation_type != 0)
        {
            float* outptr = out;
            int size = outw * 4;
            
#if __ARM_NEON
            for (int i = 0; i < size; i += 4)
            {
                float32x4_t _out = vld1q_f32(outptr + i);
                _out = activation_ps(_out, activation_type, activation_params);
                vst1q_f32(outptr + i, _out);
            }
#else
            for (int i = 0; i < size; i++)
            {
                outptr[i] = activation_ss(outptr[i], activation_type, activation_params);
            }
#endif
        }
    }

    return 0;
}

// Auto selection function
static int deconv1d_k2s2_pack_auto(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt)
{
    int elempack = bottom_blob.elempack;
    int out_elempack = top_blob.elempack;
    
    if (elempack == 4 && out_elempack == 4)
    {
        return deconv1d_k2s2_pack4_neon(bottom_blob, top_blob, weight_data, bias_data, activation_type, activation_params, opt);
    }
    else if (elempack == 1 && out_elempack == 4)
    {
        return deconv1d_k2s2_pack1to4_neon(bottom_blob, top_blob, weight_data, bias_data, activation_type, activation_params, opt);
    }
    else
    {
        // Fallback to existing optimized implementation
        return deconv1d_k2s2_auto(bottom_blob, top_blob, weight_data, bias_data, activation_type, activation_params, opt);
    }
}
