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

    // Calculate total loop iterations for prediction
    const int total_iterations = outh * w * h;
    fprintf(stderr, "deconv1d_k2s2_neon: predicted total iterations = %d (outh=%d, w=%d, h=%d)\n", 
            total_iterations, outh, w, h);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outh; p++)
    {
        Mat out = top_blob.row_range(p, 1);

        const float bias0 = bias_term ? bias[p] : 0.f;

        out.fill(bias0);

        for (int j = 0; j < w; j++)
        {
            float* outptr = (float*)out + j * 2; // stride_w = 2

            const float* kptr = kernel + 2 * h * p; // kernel_w = 2

            for (int q = 0; q < h; q++)
            {
                const float val = bottom_blob.row(q)[j];

#if __ARM_NEON
                // Load 2 kernel weights
                float32x2_t _k = vld1_f32(kptr);
                
                // Load 2 output values
                float32x2_t _out = vld1_f32(outptr);
                
                // Multiply input value with kernel weights and accumulate
                _out = vfma_n_f32(_out, _k, val);
                
                // Store back
                vst1_f32(outptr, _out);
#else
                for (int k = 0; k < 2; k++) // kernel_w = 2
                {
                    float w = kptr[k];
                    outptr[k] += val * w; // dilation_w = 1
                }
#endif // __ARM_NEON

                kptr += 2; // kernel_w = 2
            }
        }
    }

    // Apply activation
    if (activation_type != 0)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outh; p++)
        {
            Mat out = top_blob.row_range(p, 1);
            float* outptr = out;
            for (int i = 0; i < outw; i++)
            {
                outptr[i] = activation_ss(outptr[i], activation_type, activation_params);
            }
        }
    }

    return 0;
}