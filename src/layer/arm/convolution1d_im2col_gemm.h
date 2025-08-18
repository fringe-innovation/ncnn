// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// Simplified version for 1x1 convolution
static void convolution1d_im2col_input_tile_conv1x1s1d1(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk)
{
    const int elempack = bottom_blob.elempack;
    float* pp = B;

    int jj = 0;
#if __ARM_NEON
    for (; jj + 3 < max_jj; jj += 4)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)bottom_blob.channel(k / 4) + (j + jj) * 4;

            int kk = 0;
            for (; kk < max_kk / 4; kk++)
            {
                // transpose4x4
                float32x4x4_t _r0123 = vld4q_f32(p0);
                vst1q_f32(pp, _r0123.val[0]);
                vst1q_f32(pp + 4, _r0123.val[1]);
                vst1q_f32(pp + 8, _r0123.val[2]);
                vst1q_f32(pp + 12, _r0123.val[3]);
                pp += 16;
                p0 += bottom_blob.cstep * 4;
            }
        }

        if (elempack == 1)
        {
            const float* p0 = (const float*)bottom_blob.channel(k) + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                float32x4_t _r0 = vld1q_f32(p0);
                vst1q_f32(pp, _r0);
                pp += 4;
                p0 += bottom_blob.cstep;
            }
        }
    }
#endif // __ARM_NEON
    for (; jj < max_jj; jj++)
    {
#if __ARM_NEON
        if (elempack == 4)
        {
            const float* p0 = (const float*)bottom_blob.channel(k / 4) + (j + jj) * 4;

            int kk = 0;
            for (; kk < max_kk / 4; kk++)
            {
                float32x4_t _r0 = vld1q_f32(p0);
                vst1q_f32(pp, _r0);
                pp += 4;
                p0 += bottom_blob.cstep * 4;
            }
        }
#endif // __ARM_NEON

        if (elempack == 1)
        {
            const float* p0 = (const float*)bottom_blob.channel(k) + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0 += bottom_blob.cstep;
            }
        }
    }
}

static void convolution1d_im2col_gemm_transform_kernel(const Mat& kernel, Mat& AT, int inch, int outch, int kernel_w, const Option& opt)
{
    const int maxk = kernel_w;
    const int M = outch;
    const int K = inch * maxk;

    // For 1x1 convolution, we can use a simplified transformation
    if (maxk == 1)
    {
        AT = kernel.reshape(K, M);
        return;
    }

    // General case for other kernel sizes
    Mat weight_data_r2 = kernel.reshape(maxk, inch, outch);
    AT.create(K, M);

    for (int q = 0; q < outch; q++)
    {
        float* g00 = AT.row(q);
        
        for (int p = 0; p < inch; p++)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k00 = weight_data_r2.channel(q).row(p);
                g00[p * maxk + k] = k00[k];
            }
        }
    }
}

static int convolution1d_im2col_gemm(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, const Mat& bias, int kernel_w, int dilation_w, int stride_w, int nT, const Option& opt)
{
    const int maxk = kernel_w;
    const int M = top_blob.c * top_blob.elempack;
    const int N = top_blob.w;
    const int K = bottom_blob.c * bottom_blob.elempack * maxk;

    // Simple tiling strategy for now
    const int TILE_M = 4;
    const int TILE_N = 12;
    const int TILE_K = 4;

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_N = (N + TILE_N - 1) / TILE_N;
    const int nn_K = (K + TILE_K - 1) / TILE_K;

    Mat BT(TILE_K * TILE_N, nn_K, nn_N, 4u, opt.workspace_allocator);
    if (BT.empty())
        return -100;

    const int nn_NK = nn_N * nn_K;

    // im2col transformation
    #pragma omp parallel for num_threads(nT)
    for (int ppjk = 0; ppjk < nn_NK; ppjk++)
    {
        const int ppj = ppjk / nn_K;
        const int ppk = ppjk % nn_K;

        const int j = ppj * TILE_N;
        const int k = ppk * TILE_K;

        const int max_jj = (N - j) < TILE_N ? (N - j) : TILE_N;
        const int max_kk = (K - k) < TILE_K ? (K - k) : TILE_K;

        Mat BT_tile = BT.channel(ppj).row_range(ppk, 1);

        // For 1x1 convolution, use optimized path
        if (kernel_w == 1 && dilation_w == 1 && stride_w == 1)
        {
            convolution1d_im2col_input_tile_conv1x1s1d1(bottom_blob, BT_tile, j, max_jj, k, max_kk);
        }
        else
        {
            // TODO: Implement general case
            fprintf(stderr, "convolution1d_im2col_gemm: Unsupported kernel size %d\n", kernel_w);
        }
    }

    // Simple GEMM computation
    #pragma omp parallel for num_threads(nT)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;
        const int max_ii = (M - i) < TILE_M ? (M - i) : TILE_M;

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = (N - j) < TILE_N ? (N - j) : TILE_N;

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = (K - k) < TILE_K ? (K - k) : TILE_K;

                const Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);
                
                // Simplified GEMM kernel for 1x1 case
                for (int ii = 0; ii < max_ii; ii++)
                {
                    const float* pA = AT.row(i + ii) + k;
                    const float* pB = BT_tile;
                    float* pC = top_blob.channel((i + ii) / top_blob.elempack).row(0) + j * top_blob.elempack + ((i + ii) % top_blob.elempack);

                    for (int jj = 0; jj < max_jj; jj++)
                    {
                        float sum = 0.f;
                        if (bias.empty())
                        {
                            sum = 0.f;
                        }
                        else
                        {
                            sum = bias[i + ii];
                        }

                        for (int kk = 0; kk < max_kk; kk++)
                        {
                            sum += pA[kk] * pB[jj + kk * max_jj];
                        }

                        pC[jj * top_blob.elempack] = sum;
                    }
                }
            }
        }
    }

    return 0;
}