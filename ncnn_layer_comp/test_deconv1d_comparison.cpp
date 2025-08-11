// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// Test program to compare standard deconv_arm and optimized deconv_v2 implementations
// Usage: ./test_deconv1d_comparison

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#if __ARM_NEON
#include <arm_neon.h>
#endif

// Forward declarations to avoid dependency issues
namespace ncnn {
    class Mat {
    public:
        float* data;
        int w, h;
        
        Mat() : data(nullptr), w(0), h(0) {}
        Mat(int width, int height) : w(width), h(height) {
            data = (float*)malloc(w * h * sizeof(float));
            memset(data, 0, w * h * sizeof(float));
        }
        ~Mat() { if (data) free(data); }
        
        float* row(int i) { return data + i * w; }
        const float* row(int i) const { return data + i * w; }
        void fill(float val) {
            for (int i = 0; i < w * h; i++) data[i] = val;
        }
        bool empty() const { return data == nullptr; }
        void create(int width, int height) {
            if (data) free(data);
            w = width; h = height;
            data = (float*)malloc(w * h * sizeof(float));
            memset(data, 0, w * h * sizeof(float));
        }
        
        // Copy constructor and assignment operator for safety
        Mat(const Mat& other) : w(other.w), h(other.h) {
            data = (float*)malloc(w * h * sizeof(float));
            memcpy(data, other.data, w * h * sizeof(float));
        }
        Mat& operator=(const Mat& other) {
            if (this != &other) {
                if (data) free(data);
                w = other.w; h = other.h;
                data = (float*)malloc(w * h * sizeof(float));
                memcpy(data, other.data, w * h * sizeof(float));
            }
            return *this;
        }
        
        float& operator[](int i) { return data[i]; }
        const float& operator[](int i) const { return data[i]; }
    };
    
    struct Option {
        int num_threads;
        Option() : num_threads(1) {}
    };
}

using namespace ncnn;

// Include the original and optimized implementations
// Original deconvolution1d_arm function (standard version)
static int deconvolution1d_arm_standard(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data, const Mat& bias_data, int kernel_w, int stride_w, int dilation_w, int activation_type, const Mat& activation_params, const Option& opt)
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

        // Apply activation
        float* outptr = out;
        for (int i = 0; i < outw; i++)
        {
            // Simple activation functions for testing
            if (activation_type == 1) // ReLU
                outptr[i] = std::max(0.0f, outptr[i]);
            else if (activation_type == 2) // LeakyReLU
                outptr[i] = outptr[i] > 0 ? outptr[i] : outptr[i] * 0.1f;
        }
    }
    return 0;
}

// Optimized 2x2 deconvolution (from deconvolution1d_2x2_optimized.h)
#include "deconvolution1d_2x2_optimized.h"

// Test data generation
Mat generate_random_mat(int w, int h, float min_val = -1.0f, float max_val = 1.0f)
{
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducible results
    std::uniform_real_distribution<float> dist(min_val, max_val);
    
    Mat mat(w, h);
    for (int i = 0; i < h; i++)
    {
        float* ptr = mat.row(i);
        for (int j = 0; j < w; j++)
        {
            ptr[j] = dist(gen);
        }
    }
    return mat;
}

// Compare two matrices and return maximum absolute difference
float compare_mats(const Mat& mat1, const Mat& mat2, bool verbose = false)
{
    if (mat1.w != mat2.w || mat1.h != mat2.h)
    {
        std::cout << "ERROR: Matrix dimensions mismatch!" << std::endl;
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
            float diff = std::abs(ptr1[j] - ptr2[j]);
            max_diff = std::max(max_diff, diff);
            total_diff += diff;
            count++;
            
            if (verbose && diff > 1e-5f)
            {
                std::cout << "Diff at [" << i << "][" << j << "]: " 
                          << ptr1[j] << " vs " << ptr2[j] 
                          << " (diff: " << diff << ")" << std::endl;
            }
        }
    }
    
    std::cout << "Max difference: " << max_diff << std::endl;
    std::cout << "Average difference: " << (count > 0 ? total_diff / count : 0.0f) << std::endl;
    
    return max_diff;
}

// Performance benchmark
struct BenchmarkResult 
{
    double time_ms;
    std::string name;
};

BenchmarkResult benchmark_function(std::function<int()> func, const std::string& name, int iterations = 100)
{
    // Warmup
    for (int i = 0; i < 10; i++) {
        func();
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        func();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    BenchmarkResult result;
    result.time_ms = duration.count() / 1000.0 / iterations;
    result.name = name;
    
    return result;
}

// Test case structure
struct TestCase 
{
    int input_w, input_h, output_h;
    int kernel_w, stride_w, dilation_w;
    int activation_type;
    bool use_bias;
    std::string description;
};

// Run a single test case
bool run_test_case(const TestCase& test_case, bool verbose = false)
{
    std::cout << "\n=== Testing: " << test_case.description << " ===" << std::endl;
    std::cout << "Input: " << test_case.input_w << "x" << test_case.input_h 
              << ", Output: " << test_case.output_h 
              << ", Kernel: " << test_case.kernel_w
              << ", Stride: " << test_case.stride_w << std::endl;

    // Generate test data
    Mat input = generate_random_mat(test_case.input_w, test_case.input_h);
    Mat weights = generate_random_mat(test_case.kernel_w * test_case.input_h * test_case.output_h, 1);
    Mat bias = test_case.use_bias ? generate_random_mat(test_case.output_h, 1) : Mat();
    
    // Calculate output dimensions
    const int kernel_extent_w = test_case.dilation_w * (test_case.kernel_w - 1) + 1;
    int outw = (test_case.input_w - 1) * test_case.stride_w + kernel_extent_w;
    
    // Create output matrices
    Mat output_standard(outw, test_case.output_h);
    Mat output_optimized(outw, test_case.output_h);
    
    // Setup options
    Option opt;
    opt.num_threads = 1;
    
    Mat activation_params(2);
    activation_params[0] = -0.1f; // alpha for LeakyReLU
    activation_params[1] = 0.1f;  // beta
    
    bool test_passed = true;
    
    // Test standard implementation
    try {
        int ret1 = deconvolution1d_arm_standard(input, output_standard, weights, bias, 
                                               test_case.kernel_w, test_case.stride_w, test_case.dilation_w,
                                               test_case.activation_type, activation_params, opt);
        if (ret1 != 0) {
            std::cout << "ERROR: Standard implementation failed with return code: " << ret1 << std::endl;
            return false;
        }
    } catch (const std::exception& e) {
        std::cout << "ERROR: Standard implementation threw exception: " << e.what() << std::endl;
        return false;
    }
    
    // Test optimized implementation (only for 2x2 case)
    if (test_case.kernel_w == 2 && test_case.stride_w == 2 && test_case.dilation_w == 1) {
        try {
            int ret2 = deconv1d_k2s2_auto(input, output_optimized, weights, bias,
                                         test_case.activation_type, activation_params, opt);
            if (ret2 != 0) {
                std::cout << "ERROR: Optimized implementation failed with return code: " << ret2 << std::endl;
                return false;
            }
            
            // Compare results
            float max_diff = compare_mats(output_standard, output_optimized, verbose);
            
            const float tolerance = 1e-5f;
            if (max_diff > tolerance) {
                std::cout << "FAIL: Results differ by more than tolerance (" << tolerance << ")" << std::endl;
                test_passed = false;
            } else {
                std::cout << "PASS: Results match within tolerance" << std::endl;
            }
            
            // Performance comparison
            std::cout << "\nPerformance Comparison:" << std::endl;
            
            auto bench_standard = benchmark_function([&]() {
                return deconvolution1d_arm_standard(input, output_standard, weights, bias,
                                                   test_case.kernel_w, test_case.stride_w, test_case.dilation_w,
                                                   test_case.activation_type, activation_params, opt);
            }, "Standard", 50);
            
            auto bench_optimized = benchmark_function([&]() {
                return deconv1d_k2s2_auto(input, output_optimized, weights, bias,
                                         test_case.activation_type, activation_params, opt);
            }, "Optimized", 50);
            
            std::cout << "Standard:  " << std::fixed << std::setprecision(3) << bench_standard.time_ms << " ms" << std::endl;
            std::cout << "Optimized: " << std::fixed << std::setprecision(3) << bench_optimized.time_ms << " ms" << std::endl;
            
            double speedup = bench_standard.time_ms / bench_optimized.time_ms;
            std::cout << "Speedup:   " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
            
        } catch (const std::exception& e) {
            std::cout << "ERROR: Optimized implementation threw exception: " << e.what() << std::endl;
            return false;
        }
    } else {
        std::cout << "INFO: Optimized implementation only supports kernel=2, stride=2, dilation=1" << std::endl;
        std::cout << "PASS: Standard implementation completed successfully" << std::endl;
    }
    
    return test_passed;
}

int main()
{
    std::cout << "=== NCNN Deconvolution1D Implementation Comparison Test ===" << std::endl;
    
#if __ARM_NEON
    std::cout << "ARM NEON support: ENABLED" << std::endl;
#else
    std::cout << "ARM NEON support: DISABLED" << std::endl;
#endif

    // Define test cases
    std::vector<TestCase> test_cases = {
        // 2x2 optimized cases
        {9, 4, 8, 2, 2, 1, 0, true, "2x2 kernel, stride=2, no activation, with bias"},
        {9, 4, 8, 2, 2, 1, 1, true, "2x2 kernel, stride=2, ReLU activation, with bias"},
        {9, 4, 8, 2, 2, 1, 2, false, "2x2 kernel, stride=2, LeakyReLU activation, no bias"},
        {16, 8, 16, 2, 2, 1, 0, true, "2x2 kernel, stride=2, larger input, no activation"},
        {32, 16, 32, 2, 2, 1, 1, true, "2x2 kernel, stride=2, large input, ReLU activation"},
        
        // Other cases (only standard implementation)
        {9, 4, 8, 3, 1, 1, 0, true, "3x1 kernel, stride=1, no activation"},
        {9, 4, 8, 1, 1, 1, 1, false, "1x1 kernel, stride=1, ReLU activation"},
        {9, 4, 8, 4, 2, 1, 0, true, "4x2 kernel, stride=2, no activation"},
    };
    
    bool all_tests_passed = true;
    int passed_count = 0;
    int total_count = test_cases.size();
    
    // Run all test cases
    for (const auto& test_case : test_cases) {
        bool result = run_test_case(test_case, false); // Set to true for verbose output
        if (result) {
            passed_count++;
        } else {
            all_tests_passed = false;
        }
    }
    
    // Summary
    std::cout << "\n=== Test Summary ===" << std::endl;
    std::cout << "Passed: " << passed_count << "/" << total_count << std::endl;
    
    if (all_tests_passed) {
        std::cout << "All tests PASSED! ✓" << std::endl;
        return 0;
    } else {
        std::cout << "Some tests FAILED! ✗" << std::endl;
        return 1;
    }
}
