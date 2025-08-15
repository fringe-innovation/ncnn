// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "deconvolution1d_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#include "arm_usability.h"
#include "fused_activation.h"

namespace ncnn {

#include "deconvolution1d_2x2.h"
#include "deconvolution1d_2x2_optimized.h"
#include "deconvolution1d_2x2_pack4.h"

Deconvolution1D_arm::Deconvolution1D_arm()
{
#if __ARM_NEON
    support_packing = true;
#endif // __ARM_NEON
}

static int deconvolution1d_arm(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data, const Mat& bias_data, int kernel_w, int stride_w, int dilation_w, int activation_type, const Mat& activation_params, const Option& opt)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;

    const int outw = top_blob.w;
    const int outh = top_blob.h;

    const int bias_term = bias_data.empty() ? 0 : 1;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outh; p++)
    {
        Mat out = top_blob.row_range(p, 1);

        const float bias = bias_term ? bias_data[p] : 0.f;

        out.fill(bias);

        for (int j = 0; j < w; j++)
        {
            float* outptr = (float*)out + j * stride_w;

            const float* kptr = (const float*)weight_data + kernel_w * h * p;

            for (int q = 0; q < h; q++)
            {
                const float val = bottom_blob.row(q)[j];

                for (int k = 0; k < kernel_w; k++)
                {
                    float w = kptr[k];
                    outptr[k * dilation_w] += val * w;
                }

                kptr += kernel_w;
            }
        }

        {
            float* outptr = out;

            for (int i = 0; i < outw; i++)
            {
                outptr[i] = activation_ss(outptr[i], activation_type, activation_params);
            }
        }
    }

    return 0;
}

int Deconvolution1D_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    size_t elemsize = bottom_blob.elemsize;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;

    int outw = (w - 1) * stride_w + kernel_extent_w + output_pad_right;

    // Determine packing parameters
    int out_elempack = 1;
    size_t out_elemsize = elemsize;
    
#if __ARM_NEON
    if (opt.use_packing_layout && kernel_w == 2 && stride_w == 2 && dilation_w == 1)
    {
        // Use packed layout for 2x2 optimized case
        if (num_output % 4 == 0)
        {
            out_elempack = 4;
            out_elemsize = elemsize * out_elempack;
        }
    }
#endif

    Mat top_blob_bordered;
    if (pad_left > 0 || pad_right > 0 || output_w > 0)
    {
        if (out_elempack == 4)
            top_blob_bordered.create(outw, num_output / out_elempack, out_elemsize, out_elempack, opt.workspace_allocator);
        else
            top_blob_bordered.create(outw, num_output, elemsize, opt.workspace_allocator);
    }
    else
    {
        top_blob_bordered = top_blob;
        if (out_elempack == 4)
            top_blob_bordered.create(outw, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        else
            top_blob_bordered.create(outw, num_output, elemsize, opt.blob_allocator);
    }
    if (top_blob_bordered.empty())
        return -100;

    // Use optimized kernel for specific cases
    if (kernel_w == 2 && stride_w == 2 && dilation_w == 1)
    {
#if __ARM_NEON
        // Check if we should use packed optimizations
        if (opt.use_packing_layout && out_elempack == 4)
        {
            // fprintf(stderr, "deconvolution1d_arm: using deconv1d_k2s2_pack_auto (packed optimized)\n");
            int ret = deconv1d_k2s2_pack_auto(bottom_blob, top_blob_bordered, weight_data, bias_data, activation_type, activation_params, opt);
            if (ret != 0)
                return ret;
        }
        else
        {
            // fprintf(stderr, "deconvolution1d_arm: using deconv1d_k2s2_auto (optimized)\n");
            int ret = deconv1d_k2s2_auto(bottom_blob, top_blob_bordered, weight_data, bias_data, activation_type, activation_params, opt);
            if (ret != 0)
                return ret;
        }
#else
        // fprintf(stderr, "deconvolution1d_arm: using deconvolution1d_arm\n");
        int ret = deconvolution1d_arm(bottom_blob, top_blob_bordered, weight_data, bias_data, kernel_w, stride_w, dilation_w, activation_type, activation_params, opt);
        if (ret != 0)
            return ret;
#endif
    }
    else
    {
        int ret = deconvolution1d_arm(bottom_blob, top_blob_bordered, weight_data, bias_data, kernel_w, stride_w, dilation_w, activation_type, activation_params, opt);
        if (ret != 0)
            return ret;
    }

    cut_padding(top_blob_bordered, top_blob, opt);
    if (top_blob.empty())
        return -100;

    return 0;
}

int Deconvolution1D_arm::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& _weight_data = bottom_blobs[1];
    Mat& top_blob = top_blobs[0];

    const int _num_input = bottom_blob.h;
    const int _kernel_w = _weight_data.w;
    const int _num_output = _weight_data.h * 1;

    Mat weight_data_flattened;
    flatten(_weight_data, weight_data_flattened, opt);
    if (weight_data_flattened.empty())
        return -100;

    // transpose group-inch/group-outch/group-kw to group-outch/group-inch/group-kw
    Mat weight_data_transposed;
    {
        weight_data_transposed.create(_kernel_w * _num_output * _num_input / 1, 4u, opt.workspace_allocator);
        if (weight_data_transposed.empty())
            return -100;

        const int outch_g = _num_output / 1;
        const int inch_g = _num_input / 1;
        const int maxk = _kernel_w;

        for (int g = 0; g < 1; g++)
        {
            // reorder weight from inch-outch to outch-inch
            float* wg2 = (float*)weight_data_transposed + g * outch_g * inch_g * maxk;
            const float* wg = (const float*)weight_data_flattened + g * inch_g * outch_g * maxk;
            for (int i = 0; i < outch_g; i++)
            {
                for (int j = 0; j < inch_g; j++)
                {
                    for (int k = 0; k < maxk; k++)
                    {
                        wg2[(i * inch_g + j) * maxk + k] = wg[(j * outch_g + i) * maxk + k];
                    }
                }
            }
        }
    }

    Mat bias_data_flattened;
    if (bias_term)
    {
        const Mat& _bias_data = bottom_blobs[2];
        flatten(_bias_data, bias_data_flattened, opt);
        if (bias_data_flattened.empty())
            return -100;
    }

    const int w = bottom_blob.w;

    const int kernel_extent_w = dilation_w * (_kernel_w - 1) + 1;

    int outw = (w - 1) * stride_w + kernel_extent_w + output_pad_right;

    Mat top_blob_bordered;
    if (pad_left > 0 || pad_right > 0 || output_w > 0)
    {
        top_blob_bordered.create(outw, _num_output, 4u, opt.workspace_allocator);
    }
    else
    {
        top_blob_bordered = top_blob;
        top_blob_bordered.create(outw, _num_output, 4u, opt.blob_allocator);
    }
    if (top_blob_bordered.empty())
        return -100;

    // Use optimized kernel for specific cases
    if (_kernel_w == 2 && stride_w == 2 && dilation_w == 1)
    {
#if __ARM_NEON
        int ret = deconv1d_k2s2_auto(bottom_blob, top_blob_bordered, weight_data_transposed, bias_data_flattened, activation_type, activation_params, opt);
#else
        int ret = deconvolution1d_arm(bottom_blob, top_blob_bordered, weight_data_transposed, bias_data_flattened, _kernel_w, stride_w, dilation_w, activation_type, activation_params, opt);
#endif
        if (ret != 0)
            return ret;
    }
    else
    {
    int ret = deconvolution1d_arm(bottom_blob, top_blob_bordered, weight_data_transposed, bias_data_flattened, _kernel_w, stride_w, dilation_w, activation_type, activation_params, opt);
    if (ret != 0)
        return ret;
    }

    cut_padding(top_blob_bordered, top_blob, opt);
    if (top_blob.empty())
        return -100;

    return 0;
}

} // namespace ncnn
