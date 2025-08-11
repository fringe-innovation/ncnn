// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static int deconv1d_k2s2_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int outw = top_blob.w;
    const int outh = top_blob.h;

    const float* kernel = weight_data;
    const float* bias = bias_data;

    const int bias_term = bias_data.empty() ? 0 : 1;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outh; p++)
    {
        float* outptr = top_blob.row(p);
        const float bias0 = bias_term ? bias[p] : 0.f;
        const float* kptr = kernel + 2 * h * p;

        // Initialize output with bias
        for (int i = 0; i < outw; i++)
        {
            outptr[i] = bias0;
        }

#if __ARM_NEON
        // Process 4 input positions at once when possible
        int j = 0;
        for (; j + 3 < w; j += 4)
        {
            // Process 4 input positions simultaneously
            for (int q = 0; q < h; q++)
            {
                const float* inptr = bottom_blob.row(q) + j;
                float32x4_t _val = vld1q_f32(inptr);
                
                // Load kernel weights
                float32x2_t _k = vld1_f32(kptr + q * 2);
                float k0 = vget_lane_f32(_k, 0);
                float k1 = vget_lane_f32(_k, 1);
                
                // Calculate output positions
                float* out0 = outptr + j * 2;
                float* out1 = outptr + (j + 1) * 2;
                float* out2 = outptr + (j + 2) * 2;
                float* out3 = outptr + (j + 3) * 2;
                
                // Accumulate results
                out0[0] += vgetq_lane_f32(_val, 0) * k0;
                out0[1] += vgetq_lane_f32(_val, 0) * k1;
                out1[0] += vgetq_lane_f32(_val, 1) * k0;
                out1[1] += vgetq_lane_f32(_val, 1) * k1;
                out2[0] += vgetq_lane_f32(_val, 2) * k0;
                out2[1] += vgetq_lane_f32(_val, 2) * k1;
                out3[0] += vgetq_lane_f32(_val, 3) * k0;
                out3[1] += vgetq_lane_f32(_val, 3) * k1;
            }
        }
        
        // Handle remaining input positions
        for (; j < w; j++)
        {
            for (int q = 0; q < h; q++)
            {
                const float val = bottom_blob.row(q)[j];
                float* out = outptr + j * 2;
                
                float32x2_t _k = vld1_f32(kptr + q * 2);
                float32x2_t _out = vld1_f32(out);
                _out = vfma_n_f32(_out, _k, val);
                vst1_f32(out, _out);
            }
        }
#else
        // Scalar fallback
        for (int j = 0; j < w; j++)
        {
            float* out = outptr + j * 2;
            
            for (int q = 0; q < h; q++)
            {
                const float val = bottom_blob.row(q)[j];
                const float* k = kptr + q * 2;
                
                out[0] += val * k[0];
                out[1] += val * k[1];
            }
        }
#endif

        // Apply activation in the same loop to improve cache efficiency
        if (activation_type != 0)
        {
            for (int i = 0; i < outw; i++)
            {
                outptr[i] = activation_ss(outptr[i], activation_type, activation_params);
            }
        }
    }

    return 0;
}