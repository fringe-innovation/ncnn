// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// Simplified test program to compare standard deconv_arm and optimized deconv_v2
// This version includes necessary definitions to avoid external dependencies

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#if __ARM_NEON
#include <arm_neon.h>
#endif

// Minimal Mat class for testing
class SimpleMat {
public:
    float* data;
    int w, h;
    
    SimpleMat() : data(nullptr), w(0), h(0) {}
    
    SimpleMat(int width, int height) : w(width), h(height) {
        data = (float*)malloc(w * h * sizeof(float));
        memset(data, 0, w * h * sizeof(float));
    }
    
    ~SimpleMat() {
        if (data) free(data);
    }
    
    float* row(int i) { return data + i * w; }
    const float* row(int i) const { return data + i * w; }
    
    void fill(float val) {
        for (int i = 0; i < w * h; i++) {
            data[i] = val;
        }
    }
    
    bool empty() const { return data == nullptr; }
    
    void create(int width, int height) {
        if (data) free(data);
        w = width;
        h = height;
        data = (float*)malloc(w * h * sizeof(float));
        memset(data, 0, w * h * sizeof(float));
    }
};

// Simple Option class
struct SimpleOption {
    int num_threads;
    SimpleOption() : num_threads(1) {}
};

// Simple activation function
float activation_ss(float x, int activation_type, const SimpleMat& params) {
    switch (activation_type) {
        case 0: return x; // No activation
        case 1: return fmaxf(0.0f, x); // ReLU
        case 2: return x > 0 ? x : x * 0.1f; // LeakyReLU with 0.1 slope
        default: return x;
    }
}

#if __ARM_NEON
float32x4_t activation_ps(float32x4_t x, int activation_type, const SimpleMat& params) {
    switch (activation_type) {
        case 0: return x; // No activation
        case 1: return vmaxq_f32(x, vdupq_n_f32(0.0f)); // ReLU
        case 2: { // LeakyReLU
            float32x4_t zero = vdupq_n_f32(0.0f);
            float32x4_t alpha = vdupq_n_f32(0.1f);
            uint32x4_t mask = vcgtq_f32(x, zero);
            float32x4_t negative_part = vmulq_f32(x, alpha);
            return vbslq_f32(mask, x, negative_part);
        }
        default: return x;
    }
}
#endif

// Standard deconvolution implementation
int deconvolution1d_arm_standard(const SimpleMat& bottom_blob, SimpleMat& top_blob, 
                                const SimpleMat& weight_data, const SimpleMat& bias_data, 
                                int kernel_w, int stride_w, int dilation_w, 
                                int activation_type, const SimpleMat& activation_params, 
                                const SimpleOption& opt)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int bias_term = bias_data.empty() ? 0 : 1;

    for (int p = 0; p < outh; p++)
    {
        float* outptr = top_blob.row(p);
        const float bias = bias_term ? bias_data.data[p] : 0.f;
        
        // Initialize with bias
        for (int i = 0; i < outw; i++) {
            outptr[i] = bias;
        }

        for (int j = 0; j < w; j++)
        {
            float* out_base = outptr + j * stride_w;
            const float* kptr = weight_data.data + kernel_w * h * p;

            for (int q = 0; q < h; q++)
            {
                const float val = bottom_blob.row(q)[j];
                for (int k = 0; k < kernel_w; k++)
                {
                    float weight = kptr[k];
                    out_base[k * dilation_w] += val * weight;
                }
                kptr += kernel_w;
            }
        }

        // Apply activation
        for (int i = 0; i < outw; i++)
        {
            outptr[i] = activation_ss(outptr[i], activation_type, activation_params);
        }
    }
    return 0;
}

// Optimized 2x2 deconvolution with NEON
int deconv1d_k2s2_optimized(const SimpleMat& bottom_blob, SimpleMat& top_blob, 
                           const SimpleMat& weight_data, const SimpleMat& bias_data, 
                           int activation_type, const SimpleMat& activation_params, 
                           const SimpleOption& opt)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const float* kernel = weight_data.data;
    const float* bias = bias_data.data;
    const int bias_term = bias_data.empty() ? 0 : 1;

    for (int p = 0; p < outh; p++)
    {
        float* outptr = top_blob.row(p);
        const float bias0 = bias_term ? bias[p] : 0.f;
        const float* kptr = kernel + 2 * h * p;

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
#if __ARM_NEON
        for (int q = 0; q < h; q++)
        {
            const float* inptr = bottom_blob.row(q);
            float32x2_t _k = vld1_f32(kptr + q * 2);
            float k0 = vget_lane_f32(_k, 0);
            float k1 = vget_lane_f32(_k, 1);
            
            // Process 4 input positions at once
            int j = 0;
            for (; j + 3 < w; j += 4)
            {
                float32x4_t _val = vld1q_f32(inptr + j);
                float* out_base = outptr + j * 2;
                
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
        // Scalar version
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

        // Apply activation
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

// Test data generation
void generate_random_data(SimpleMat& mat, float min_val = -1.0f, float max_val = 1.0f)
{
    srand(42); // Fixed seed for reproducible results
    for (int i = 0; i < mat.w * mat.h; i++)
    {
        float rand_val = (float)rand() / RAND_MAX;
        mat.data[i] = min_val + rand_val * (max_val - min_val);
    }
}

// Compare two matrices
float compare_matrices(const SimpleMat& mat1, const SimpleMat& mat2, bool verbose = false)
{
    if (mat1.w != mat2.w || mat1.h != mat2.h)
    {
        printf("ERROR: Matrix dimensions mismatch!\n");
        return -1.0f;
    }
    
    float max_diff = 0.0f;
    float total_diff = 0.0f;
    int count = 0;
    
    for (int i = 0; i < mat1.h; i++)
    {
        const float* ptr1 = mat1.row(i);
        const float* ptr2 = mat2.row(i);
        
        for (int j = 0; j < mat1.w; j++)
        {
            float diff = fabsf(ptr1[j] - ptr2[j]);
            max_diff = fmaxf(max_diff, diff);
            total_diff += diff;
            count++;
            
            if (verbose && diff > 1e-5f)
            {
                printf("Diff at [%d][%d]: %f vs %f (diff: %f)\n", 
                       i, j, ptr1[j], ptr2[j], diff);
            }
        }
    }
    
    printf("Max difference: %.9f\n", max_diff);
    printf("Average difference: %.9f\n", count > 0 ? total_diff / count : 0.0f);
    
    return max_diff;
}

// Timing utilities
double get_time_ms()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

// Test case structure
struct TestCase 
{
    int input_w, input_h, output_h;
    int kernel_w, stride_w, dilation_w;
    int activation_type;
    int use_bias;
    const char* description;
};

// Run a single test case
int run_test_case(const TestCase& test_case, int verbose)
{
    printf("\n=== Testing: %s ===\n", test_case.description);
    printf("Input: %dx%d, Output: %d, Kernel: %d, Stride: %d\n", 
           test_case.input_w, test_case.input_h, test_case.output_h, 
           test_case.kernel_w, test_case.stride_w);

    // Generate test data
    SimpleMat input(test_case.input_w, test_case.input_h);
    SimpleMat weights(test_case.kernel_w * test_case.input_h * test_case.output_h, 1);
    SimpleMat bias;
    if (test_case.use_bias) {
        bias.create(test_case.output_h, 1);
    }
    
    generate_random_data(input);
    generate_random_data(weights);
    if (test_case.use_bias) {
        generate_random_data(bias, -0.5f, 0.5f);
    }
    
    // Calculate output dimensions
    const int kernel_extent_w = test_case.dilation_w * (test_case.kernel_w - 1) + 1;
    int outw = (test_case.input_w - 1) * test_case.stride_w + kernel_extent_w;
    
    // Create output matrices
    SimpleMat output_standard(outw, test_case.output_h);
    SimpleMat output_optimized(outw, test_case.output_h);
    
    SimpleOption opt;
    SimpleMat activation_params(2, 1);
    activation_params.data[0] = -0.1f; // alpha
    activation_params.data[1] = 0.1f;  // beta
    
    // Test standard implementation
    int ret1 = deconvolution1d_arm_standard(input, output_standard, weights, bias,
                                           test_case.kernel_w, test_case.stride_w, test_case.dilation_w,
                                           test_case.activation_type, activation_params, opt);
    if (ret1 != 0) {
        printf("ERROR: Standard implementation failed with return code: %d\n", ret1);
        return 0;
    }
    
    int test_passed = 1;
    
    // Test optimized implementation (only for 2x2 case)
    if (test_case.kernel_w == 2 && test_case.stride_w == 2 && test_case.dilation_w == 1) {
        int ret2 = deconv1d_k2s2_optimized(input, output_optimized, weights, bias,
                                          test_case.activation_type, activation_params, opt);
        if (ret2 != 0) {
            printf("ERROR: Optimized implementation failed with return code: %d\n", ret2);
            return 0;
        }
        
        // Compare results
        float max_diff = compare_matrices(output_standard, output_optimized, verbose);
        
        const float tolerance = 1e-5f;
        if (max_diff > tolerance) {
            printf("FAIL: Results differ by more than tolerance (%.9f)\n", tolerance);
            test_passed = 0;
        } else {
            printf("PASS: Results match within tolerance\n");
        }
        
        // Performance comparison
        printf("\nPerformance Comparison:\n");
        
        const int iterations = 100;
        
        // Benchmark standard implementation
        double start_time = get_time_ms();
        for (int i = 0; i < iterations; i++) {
            deconvolution1d_arm_standard(input, output_standard, weights, bias,
                                        test_case.kernel_w, test_case.stride_w, test_case.dilation_w,
                                        test_case.activation_type, activation_params, opt);
        }
        double standard_time = (get_time_ms() - start_time) / iterations;
        
        // Benchmark optimized implementation
        start_time = get_time_ms();
        for (int i = 0; i < iterations; i++) {
            deconv1d_k2s2_optimized(input, output_optimized, weights, bias,
                                   test_case.activation_type, activation_params, opt);
        }
        double optimized_time = (get_time_ms() - start_time) / iterations;
        
        printf("Standard:  %.3f ms\n", standard_time);
        printf("Optimized: %.3f ms\n", optimized_time);
        printf("Speedup:   %.2fx\n", standard_time / optimized_time);
        
    } else {
        printf("INFO: Optimized implementation only supports kernel=2, stride=2, dilation=1\n");
        printf("PASS: Standard implementation completed successfully\n");
    }
    
    return test_passed;
}

int main()
{
    printf("=== NCNN Deconvolution1D Implementation Comparison Test ===\n");
    
#if __ARM_NEON
    printf("ARM NEON support: ENABLED\n");
#else
    printf("ARM NEON support: DISABLED\n");
#endif

    // Define test cases
    TestCase test_cases[] = {
        // 2x2 optimized cases
        {9, 4, 8, 2, 2, 1, 0, 1, "2x2 kernel, stride=2, no activation, with bias"},
        {9, 4, 8, 2, 2, 1, 1, 1, "2x2 kernel, stride=2, ReLU activation, with bias"},
        {9, 4, 8, 2, 2, 1, 2, 0, "2x2 kernel, stride=2, LeakyReLU activation, no bias"},
        {16, 8, 16, 2, 2, 1, 0, 1, "2x2 kernel, stride=2, larger input, no activation"},
        {32, 16, 32, 2, 2, 1, 1, 1, "2x2 kernel, stride=2, large input, ReLU activation"},
        
        // Other cases (only standard implementation)
        {9, 4, 8, 3, 1, 1, 0, 1, "3x1 kernel, stride=1, no activation"},
        {9, 4, 8, 1, 1, 1, 1, 0, "1x1 kernel, stride=1, ReLU activation"},
        {9, 4, 8, 4, 2, 1, 0, 1, "4x2 kernel, stride=2, no activation"},
    };
    
    int total_tests = sizeof(test_cases) / sizeof(TestCase);
    int passed_count = 0;
    
    // Run all test cases
    for (int i = 0; i < total_tests; i++) {
        int result = run_test_case(test_cases[i], 0); // Set to 1 for verbose output
        if (result) {
            passed_count++;
        }
    }
    
    // Summary
    printf("\n=== Test Summary ===\n");
    printf("Passed: %d/%d\n", passed_count, total_tests);
    
    if (passed_count == total_tests) {
        printf("All tests PASSED! ✓\n");
        return 0;
    } else {
        printf("Some tests FAILED! ✗\n");
        return 1;
    }
}
