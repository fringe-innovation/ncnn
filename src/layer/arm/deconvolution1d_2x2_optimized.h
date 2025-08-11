// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// Optimized 2x2 deconvolution implementation with multiple optimization strategies

#if __ARM_NEON
#include <arm_neon.h>
#include "arm_activation.h"
#endif

// Strategy 1: SIMD + Loop Unrolling + Better Memory Access
static int deconv1d_k2s2_neon_v2(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt)
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

        // Initialize with bias
        if (bias_term)
        {
#if __ARM_NEON
            float32x4_t _bias = vdupq_n_f32(bias0);
            int i = 0;
            for (; i + 3 < outw; i += 4)
            {
                vst1q_f32(outptr + i, _bias);
            }
            for (; i < outw; i++)
            {
                outptr[i] = bias0;
            }
#else
            for (int i = 0; i < outw; i++)
            {
                outptr[i] = bias0;
            }
#endif
        }
        else
        {
            memset(outptr, 0, outw * sizeof(float));
        }

        // Main computation
        const float* kptr = kernel + 2 * h * p;

#if __ARM_NEON
        // Strategy: Process multiple input channels and positions simultaneously
        for (int q = 0; q < h; q++)
        {
            const float* inptr = bottom_blob.row(q);
            float32x2_t _k = vld1_f32(kptr + q * 2);
            float k0 = vget_lane_f32(_k, 0);
            float k1 = vget_lane_f32(_k, 1);
            
            // Unroll loop for better performance
            int j = 0;
            for (; j + 3 < w; j += 4)
            {
                // Load 4 input values
                float32x4_t _val = vld1q_f32(inptr + j);
                
                // Calculate output positions (stride=2)
                float* out_base = outptr + j * 2;
                
                // Process 4 positions
                float val0 = vgetq_lane_f32(_val, 0);
                float val1 = vgetq_lane_f32(_val, 1);
                float val2 = vgetq_lane_f32(_val, 2);
                float val3 = vgetq_lane_f32(_val, 3);
                
                out_base[0] += val0 * k0;
                out_base[1] += val0 * k1;
                out_base[2] += val1 * k0;
                out_base[3] += val1 * k1;
                out_base[4] += val2 * k0;
                out_base[5] += val2 * k1;
                out_base[6] += val3 * k0;
                out_base[7] += val3 * k1;
            }
            
            // Handle remaining positions
            for (; j < w; j++)
            {
                float val = inptr[j];
                float* out = outptr + j * 2;
                out[0] += val * k0;
                out[1] += val * k1;
            }
        }
#else
        // Scalar version with better loop structure
        for (int q = 0; q < h; q++)
        {
            const float* inptr = bottom_blob.row(q);
            const float* k = kptr + q * 2;
            float k0 = k[0];
            float k1 = k[1];
            
            for (int j = 0; j < w; j++)
            {
                float val = inptr[j];
                float* out = outptr + j * 2;
                out[0] += val * k0;
                out[1] += val * k1;
            }
        }
#endif

        // Apply activation in the same loop
        if (activation_type != 0)
        {
#if __ARM_NEON
            int i = 0;
            for (; i + 3 < outw; i += 4)
            {
                float32x4_t _out = vld1q_f32(outptr + i);
                _out = activation_ps(_out, activation_type, activation_params);
                vst1q_f32(outptr + i, _out);
            }
            for (; i < outw; i++)
            {
                outptr[i] = activation_ss(outptr[i], activation_type, activation_params);
            }
#else
            for (int i = 0; i < outw; i++)
            {
                outptr[i] = activation_ss(outptr[i], activation_type, activation_params);
            }
#endif
        }
    }

    return 0;
}

// Strategy 2: Block-based optimization for better cache utilization
static int deconv1d_k2s2_neon_blocked(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int outw = top_blob.w;
    const int outh = top_blob.h;

    const float* kernel = weight_data;
    const float* bias = bias_data;
    const int bias_term = bias_data.empty() ? 0 : 1;

    // Block size for cache optimization
    const int block_size = 64; // Adjust based on cache size

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outh; p++)
    {
        float* outptr = top_blob.row(p);
        const float bias0 = bias_term ? bias[p] : 0.f;

        // Initialize output
        for (int i = 0; i < outw; i++)
        {
            outptr[i] = bias0;
        }

        const float* kptr = kernel + 2 * h * p;

        // Process in blocks for better cache locality
        for (int jb = 0; jb < w; jb += block_size)
        {
            int je = (jb + block_size < w) ? (jb + block_size) : w;
            
            for (int q = 0; q < h; q++)
            {
                const float* inptr = bottom_blob.row(q) + jb;
                const float* k = kptr + q * 2;
                float k0 = k[0];
                float k1 = k[1];

#if __ARM_NEON
                int j = 0;
                int block_w = je - jb;
                for (; j + 3 < block_w; j += 4)
                {
                    float32x4_t _val = vld1q_f32(inptr + j);
                    float* out_base = outptr + (jb + j) * 2;
                    
                    // Vectorized accumulation
                    float val0 = vgetq_lane_f32(_val, 0);
                    float val1 = vgetq_lane_f32(_val, 1);
                    float val2 = vgetq_lane_f32(_val, 2);
                    float val3 = vgetq_lane_f32(_val, 3);
                    
                    out_base[0] += val0 * k0;
                    out_base[1] += val0 * k1;
                    out_base[2] += val1 * k0;
                    out_base[3] += val1 * k1;
                    out_base[4] += val2 * k0;
                    out_base[5] += val2 * k1;
                    out_base[6] += val3 * k0;
                    out_base[7] += val3 * k1;
                }
                
                for (; j < block_w; j++)
                {
                    float val = inptr[j];
                    float* out = outptr + (jb + j) * 2;
                    out[0] += val * k0;
                    out[1] += val * k1;
                }
#else
                for (int j = 0; j < je - jb; j++)
                {
                    float val = inptr[j];
                    float* out = outptr + (jb + j) * 2;
                    out[0] += val * k0;
                    out[1] += val * k1;
                }
#endif
            }
        }

        // Apply activation
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

// Strategy 3: Choose best implementation based on data size
static int deconv1d_k2s2_auto(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    
    // Choose strategy based on problem size
    const int total_ops = w * h * top_blob.h;
    
    return deconv1d_k2s2_neon_v2(bottom_blob, top_blob, weight_data, bias_data, activation_type, activation_params, opt);

    // if (total_ops > 10000) // Large problem - use blocked version
    // {
    //     return deconv1d_k2s2_neon_blocked(bottom_blob, top_blob, weight_data, bias_data, activation_type, activation_params, opt);
    // }
    // else // Small to medium problem - use optimized version
    // {
    //     return deconv1d_k2s2_neon_v2(bottom_blob, top_blob, weight_data, bias_data, activation_type, activation_params, opt);
    // }
}
