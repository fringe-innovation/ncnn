// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// General 1D im2col transformation
static void convolution1d_im2col_input_tile(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int kernel_w, int dilation_w, int stride_w)
{
    const int w = bottom_blob.w;
    const int elempack = bottom_blob.elempack;
    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int outw = (w - kernel_extent_w) / stride_w + 1;
    const int maxk = kernel_w;

    float* pp = B;

    int jj = 0;
#if __ARM_NEON
    for (; jj + 3 < max_jj; jj += 4)
    {
        int dx0 = (j + jj) % outw;
        int dx1 = (j + jj + 1) % outw;
        int dx2 = (j + jj + 2) % outw;
        int dx3 = (j + jj + 3) % outw;

        int kk = 0;
        for (; kk < max_kk / elempack; kk++)
        {
            int p = (k / elempack + kk) / maxk;
            int v = (k / elempack + kk) % maxk;

            const Mat img = bottom_blob.channel(p);

            int x0 = stride_w * dx0 + dilation_w * v;
            int x1 = stride_w * dx1 + dilation_w * v;
            int x2 = stride_w * dx2 + dilation_w * v;
            int x3 = stride_w * dx3 + dilation_w * v;

            const float* sptr0 = img.row(0) + x0 * elempack;
            const float* sptr1 = img.row(0) + x1 * elempack;
            const float* sptr2 = img.row(0) + x2 * elempack;
            const float* sptr3 = img.row(0) + x3 * elempack;

            if (elempack == 4)
            {
                float32x4_t _r0 = vld1q_f32(sptr0);
                float32x4_t _r1 = vld1q_f32(sptr1);
                float32x4_t _r2 = vld1q_f32(sptr2);
                float32x4_t _r3 = vld1q_f32(sptr3);
                
                // transpose 4x4
                float32x4x2_t _r01 = vtrnq_f32(_r0, _r1);
                float32x4x2_t _r23 = vtrnq_f32(_r2, _r3);
                
                float32x2_t _r01_low = vget_low_f32(_r01.val[0]);
                float32x2_t _r01_high = vget_high_f32(_r01.val[0]);
                float32x2_t _r23_low = vget_low_f32(_r23.val[0]);
                float32x2_t _r23_high = vget_high_f32(_r23.val[0]);
                
                vst1_f32(pp, _r01_low);
                vst1_f32(pp + 2, _r23_low);
                vst1_f32(pp + 4, _r01_high);
                vst1_f32(pp + 6, _r23_high);
                pp += 8;
                
                _r01_low = vget_low_f32(_r01.val[1]);
                _r01_high = vget_high_f32(_r01.val[1]);
                _r23_low = vget_low_f32(_r23.val[1]);
                _r23_high = vget_high_f32(_r23.val[1]);
                
                vst1_f32(pp, _r01_low);
                vst1_f32(pp + 2, _r23_low);
                vst1_f32(pp + 4, _r01_high);
                vst1_f32(pp + 6, _r23_high);
                pp += 8;
            }

            if (elempack == 1)
            {
                pp[0] = sptr0[0];
                pp[1] = sptr1[0];
                pp[2] = sptr2[0];
                pp[3] = sptr3[0];
                pp += 4;
            }
        }
    }
#endif // __ARM_NEON
    for (; jj + 1 < max_jj; jj += 2)
    {
        int dx0 = (j + jj) % outw;
        int dx1 = (j + jj + 1) % outw;

        int kk = 0;
        for (; kk < max_kk / elempack; kk++)
        {
            int p = (k / elempack + kk) / maxk;
            int v = (k / elempack + kk) % maxk;

            const Mat img = bottom_blob.channel(p);

            int x0 = stride_w * dx0 + dilation_w * v;
            int x1 = stride_w * dx1 + dilation_w * v;

            const float* sptr0 = img.row(0) + x0 * elempack;
            const float* sptr1 = img.row(0) + x1 * elempack;

#if __ARM_NEON
            if (elempack == 4)
            {
                float32x4_t _r0 = vld1q_f32(sptr0);
                float32x4_t _r1 = vld1q_f32(sptr1);
                
                float32x4x2_t _r01 = vtrnq_f32(_r0, _r1);
                vst1q_f32(pp, _r01.val[0]);
                vst1q_f32(pp + 4, _r01.val[1]);
                pp += 8;
            }
#endif // __ARM_NEON

            if (elempack == 1)
            {
                pp[0] = sptr0[0];
                pp[1] = sptr1[0];
                pp += 2;
            }
        }
    }
    for (; jj < max_jj; jj++)
    {
        int dx = (j + jj) % outw;

        int kk = 0;
        for (; kk < max_kk / elempack; kk++)
        {
            int p = (k / elempack + kk) / maxk;
            int v = (k / elempack + kk) % maxk;

            const Mat img = bottom_blob.channel(p);

            int x = stride_w * dx + dilation_w * v;

            const float* sptr = img.row(0) + x * elempack;

#if __ARM_NEON
            if (elempack == 4)
            {
                float32x4_t _r0 = vld1q_f32(sptr);
                vst1q_f32(pp, _r0);
                pp += 4;
            }
#endif // __ARM_NEON

            if (elempack == 1)
            {
                pp[0] = sptr[0];
                pp += 1;
            }
        }
    }
}

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

    // Use simplified optimal tiling for 1D case
    int TILE_M = 4;
    int TILE_K = 4;
    
#if __ARM_NEON
    if (opt.use_packing_layout)
    {
        TILE_M = 8;
        TILE_K = 4;
    }
#endif

    // For 1x1 convolution, we can use a simplified transformation
    if (maxk == 1)
    {
        AT = kernel.reshape(K, M);
        return;
    }

    // General case for other kernel sizes  
    Mat weight_data_r2 = kernel.reshape(maxk, inch, outch);
    
    int elempack = 1;
#if __ARM_NEON
    if (opt.use_packing_layout)
    {
        elempack = inch % 4 == 0 ? 4 : 1;
    }
#endif

    // Transform kernel to GEMM format: outch x (inch * maxk)
    Mat A_data;
    A_data.create(maxk * inch, outch);

    for (int q = 0; q < outch; q++)
    {
        float* g00 = A_data.row(q);
        
        for (int p = 0; p + (elempack - 1) < inch; p += elempack)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < elempack; i++)
                {
                    const float* k00 = weight_data_r2.channel(q).row(p + i);
                    g00[0] = k00[k];
                    g00++;
                }
            }
        }
    }

    // Create tiled AT matrix
    const int nn_M = (M + TILE_M - 1) / TILE_M;
    AT.create(TILE_K * TILE_M, (K + TILE_K - 1) / TILE_K, nn_M);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;
        const int max_ii = std::min((M - i), TILE_M);

        for (int k = 0; k < K; k += TILE_K)
        {
            const int max_kk = std::min((K - k), TILE_K);

            Mat AT_tile = AT.channel(ppj).row_range(k / TILE_K, 1);
            
            // Pack A tile
            float* pp = AT_tile;
            for (int ii = 0; ii < max_ii; ii++)
            {
                const float* p0 = A_data.row(i + ii) + k;
                for (int kk = 0; kk < max_kk; kk++)
                {
                    pp[ii + kk * TILE_M] = p0[kk];
                }
            }
            // Pad remaining elements
            for (int ii = max_ii; ii < TILE_M; ii++)
            {
                for (int kk = 0; kk < max_kk; kk++)
                {
                    pp[ii + kk * TILE_M] = 0.f;
                }
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

    // Use improved tiling strategy
    int TILE_M = 4;
    int TILE_N = 12;
    int TILE_K = 4;
    
#if __ARM_NEON
    if (opt.use_packing_layout)
    {
        TILE_M = 8;
        TILE_N = 16;
        TILE_K = 4;
    }
#endif

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

        const int max_jj = std::min((N - j), TILE_N);
        const int max_kk = std::min((K - k), TILE_K);

        Mat BT_tile = BT.channel(ppj).row_range(ppk, 1);

        // For 1x1 convolution, use optimized path
        if (kernel_w == 1 && dilation_w == 1 && stride_w == 1)
        {
            convolution1d_im2col_input_tile_conv1x1s1d1(bottom_blob, BT_tile, j, max_jj, k, max_kk);
        }
        else
        {
            // General case for k3s1 and other kernel sizes
            convolution1d_im2col_input_tile(bottom_blob, BT_tile, j, max_jj, k, max_kk, kernel_w, dilation_w, stride_w);
        }
    }

    // Create temporary workspace for accumulation if needed
    Mat topT_tileX;
    if (K > TILE_K)
    {
        topT_tileX.create(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);
        if (topT_tileX.empty())
            return -100;
    }

    // GEMM computation with proper tiling
    #pragma omp parallel for num_threads(nT)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;
        const int max_ii = std::min((M - i), TILE_M);
        
        Mat topT_tile;
        if (K > TILE_K)
            topT_tile = topT_tileX.channel(get_omp_thread_num());

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min((N - j), TILE_N);

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min((K - k), TILE_K);

                const Mat AT_tile = AT.channel(ppj).row_range(k / TILE_K, 1);
                const Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);
                
                bool k_end = k + TILE_K >= K;

                // Optimized GEMM kernel
                const float* pA = AT_tile;
                const float* pB = BT_tile;
                
                for (int ii = 0; ii < max_ii; ii++)
                {
                    const int out_ch = (i + ii) / top_blob.elempack;
                    const int out_elem = (i + ii) % top_blob.elempack;
                    float* pC = top_blob.channel(out_ch).row(0) + j * top_blob.elempack + out_elem;

                    for (int jj = 0; jj < max_jj; jj++)
                    {
                        float sum = 0.f;
                        
                        // Initialize with bias or previous accumulation
                        if (k == 0)
                        {
                            if (!bias.empty())
                                sum = bias[i + ii];
                            else
                                sum = 0.f;
                        }
                        else
                        {
                            sum = pC[jj * top_blob.elempack];
                        }

                        // Compute dot product
                        const float* pA_kk = pA + ii;
                        const float* pB_jj = pB + jj;
                        
                        for (int kk = 0; kk < max_kk; kk++)
                        {
                            sum += pA_kk[kk * TILE_M] * pB_jj[kk * max_jj];
                        }

                        pC[jj * top_blob.elempack] = sum;
                    }
                }
            }
        }
    }

    return 0;
}