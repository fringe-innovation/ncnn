// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// Test program to compare standard deconv1d and packed deconv1d implementations
// Usage: ./test_deconv1d_pack_comparison

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <vector>

#if __ARM_NEON
#include <arm_neon.h>
#endif

// Forward declarations to avoid dependency issues
namespace ncnn {
    class Mat {
    public:
        float* data;
        int w, h, c;
        int elempack;
        size_t elemsize;
        
        Mat() : data(nullptr), w(0), h(0), c(0), elempack(1), elemsize(4) {}
        
        Mat(int width, int channels, size_t element_size = 4, int element_pack = 1) 
            : w(width), h(1), c(channels), elempack(element_pack), elemsize(element_size) {
            int total_size = w * c * elemsize;
            data = (float*)malloc(total_size);
            memset(data, 0, total_size);
        }
        
        Mat(int width, int channels, size_t element_size, int element_pack, void* allocator) 
            : w(width), h(1), c(channels), elempack(element_pack), elemsize(element_size) {
            int total_size = w * c * elemsize;
            data = (float*)malloc(total_size);
            memset(data, 0, total_size);
        }
        
        ~Mat() { if (data) free(data); }
        
        float* row(int i) { return data + i * w * elempack; }
        const float* row(int i) const { return data + i * w * elempack; }
        
        Mat channel(int c_idx) {
            Mat result;
            result.data = data + c_idx * w * elempack;
            result.w = w;
            result.h = h;
            result.c = 1;
            result.elempack = elempack;
            result.elemsize = elemsize;
            return result;
        }
        
        void fill(float val) {
            int total_elements = w * c * elempack;
            for (int i = 0; i < total_elements; i++) data[i] = val;
        }
        
#if __ARM_NEON
        void fill(float32x4_t val) {
            if (elempack == 4) {
                int total_vectors = w * c;
                for (int i = 0; i < total_vectors; i++) {
                    vst1q_f32(data + i * 4, val);
                }
            } else {
                fill(vgetq_lane_f32(val, 0));
            }
        }
#endif
        
        bool empty() const { return data == nullptr; }
        
        void create(int width, int channels, size_t element_size = 4, void* allocator = nullptr) {
            if (data) free(data);
            w = width; h = 1; c = channels;
            elempack = 1; elemsize = element_size;
            int total_size = w * c * elemsize;
            data = (float*)malloc(total_size);
            memset(data, 0, total_size);
        }
        
        void create(int width, int channels, size_t element_size, int element_pack, void* allocator = nullptr) {
            if (data) free(data);
            w = width; h = 1; c = channels;
            elempack = element_pack; elemsize = element_size;
            int total_size = w * c * elemsize;
            data = (float*)malloc(total_size);
            memset(data, 0, total_size);
        }
        
        // Copy constructor and assignment operator for safety
        Mat(const Mat& other) : w(other.w), h(other.h), c(other.c), elempack(other.elempack), elemsize(other.elemsize) {
            int total_size = w * c * elemsize;
            data = (float*)malloc(total_size);
            memcpy(data, other.data, total_size);
        }
        
        Mat& operator=(const Mat& other) {
            if (this != &other) {
                if (data) free(data);
                w = other.w; h = other.h; c = other.c;
                elempack = other.elempack; elemsize = other.elemsize;
                int total_size = w * c * elemsize;
                data = (float*)malloc(total_size);
                memcpy(data, other.data, total_size);
            }
            return *this;
        }
        
        float& operator[](int i) { return data[i]; }
        const float& operator[](int i) const { return data[i]; }
    };
    
    struct Option {
        int num_threads;
        bool use_packing_layout;
        void* workspace_allocator;
        void* blob_allocator;
        
        Option() : num_threads(1), use_packing_layout(true), workspace_allocator(nullptr), blob_allocator(nullptr) {}
    };
}

using namespace ncnn;

// Dummy activation functions
static float activation_ss(float x, int activation_type, const Mat& activation_params) {
    switch (activation_type) {
        case 0: return x; // No activation
        case 1: return x > 0 ? x : 0; // ReLU
        case 2: return x > 0 ? x : 0.1f * x; // LeakyReLU
        default: return x;
    }
}

#if __ARM_NEON
static float32x4_t activation_ps(float32x4_t x, int activation_type, const Mat& activation_params) {
    switch (activation_type) {
        case 0: return x; // No activation
        case 1: return vmaxq_f32(x, vdupq_n_f32(0.0f)); // ReLU
        case 2: { // LeakyReLU
            float32x4_t zeros = vdupq_n_f32(0.0f);
            float32x4_t alpha = vdupq_n_f32(0.1f);
            uint32x4_t mask = vcgtq_f32(x, zeros);
            float32x4_t negative_part = vmulq_f32(x, alpha);
            return vbslq_f32(mask, x, negative_part);
        }
        default: return x;
    }
}
#endif

// Include the original and optimized implementations
#include "../src/layer/arm/deconvolution1d_2x2.h"
#include "../src/layer/arm/deconvolution1d_2x2_optimized.h"
#include "../src/layer/arm/deconvolution1d_2x2_pack4.h"

// Standard deconvolution1d implementation (non-packed)
static int deconvolution1d_standard(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data, const Mat& bias_data, int kernel_w, int stride_w, int dilation_w, int activation_type, const Mat& activation_params, const Option& opt)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.c; // for 1D, height is number of channels
    const int outw = top_blob.w;
    const int outh = top_blob.c;
    const int bias_term = bias_data.empty() ? 0 : 1;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outh; p++)
    {
        float* outptr = top_blob.row(p);
        const float bias = bias_term ? bias_data[p] : 0.f;
        
        // Initialize with bias
        for (int i = 0; i < outw; i++) {
            outptr[i] = bias;
        }

        for (int j = 0; j < w; j++)
        {
            float* out = outptr + j * stride_w;
            const float* kptr = weight_data.data + kernel_w * h * p;

            for (int q = 0; q < h; q++)
            {
                const float val = bottom_blob.row(q)[j];
                for (int k = 0; k < kernel_w; k++)
                {
                    float weight = kptr[k];
                    out[k * dilation_w] += val * weight;
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

// Transform weight data for packed format
static void transform_weight_for_pack4(const Mat& weight_standard, Mat& weight_pack4, int kernel_w, int num_input, int num_output, bool pack4_input, bool pack4_output)
{
    if (pack4_input && pack4_output) {
        // pack4 input to pack4 output: outch/4 - inch/4 - kw - 4o - 4i (total: 4*4*kw per group)
        weight_pack4.create(kernel_w * 4 * 4, num_input / 4, num_output / 4);
        
        for (int oc_group = 0; oc_group < num_output / 4; oc_group++) {
            for (int ic_group = 0; ic_group < num_input / 4; ic_group++) {
                float* dst = weight_pack4.row(ic_group) + oc_group * kernel_w * 16;
                
                for (int k = 0; k < kernel_w; k++) {
                    for (int ic = 0; ic < 4; ic++) {
                        for (int oc = 0; oc < 4; oc++) {
                            int src_ic = ic_group * 4 + ic;
                            int src_oc = oc_group * 4 + oc;
                            int src_idx = src_oc * num_input * kernel_w + src_ic * kernel_w + k;
                            int dst_idx = k * 16 + ic * 4 + oc;
                            dst[dst_idx] = weight_standard.data[src_idx];
                        }
                    }
                }
            }
        }
    } else if (!pack4_input && pack4_output) {
        // pack1 input to pack4 output: outch/4 - inch - kw - 4o
        weight_pack4.create(kernel_w * 4, num_input, num_output / 4);
        
        for (int oc_group = 0; oc_group < num_output / 4; oc_group++) {
            for (int ic = 0; ic < num_input; ic++) {
                float* dst = weight_pack4.row(ic) + oc_group * kernel_w * 4;
                
                for (int k = 0; k < kernel_w; k++) {
                    for (int oc = 0; oc < 4; oc++) {
                        int src_oc = oc_group * 4 + oc;
                        int src_idx = src_oc * num_input * kernel_w + ic * kernel_w + k;
                        int dst_idx = k * 4 + oc;
                        dst[dst_idx] = weight_standard.data[src_idx];
                    }
                }
            }
        }
    }
}

// Transform bias for packed format
static void transform_bias_for_pack4(const Mat& bias_standard, Mat& bias_pack4, int num_output, bool pack4_output)
{
    if (pack4_output) {
        bias_pack4.create(num_output, 1, 4, 4); // 4-element packs
        
        for (int oc_group = 0; oc_group < num_output / 4; oc_group++) {
            float* dst = bias_pack4.data + oc_group * 4;
            for (int oc = 0; oc < 4; oc++) {
                dst[oc] = bias_standard.data[oc_group * 4 + oc];
            }
        }
    } else {
        bias_pack4 = bias_standard;
    }
}

// Convert between packed and unpacked formats
static void convert_pack1_to_pack4(const Mat& src, Mat& dst)
{
    int total_channels = src.c;
    if (total_channels % 4 != 0) return; // Can't pack
    
    dst.create(src.w, total_channels / 4, 16, 4); // 4 elements per pack
    
    for (int c_group = 0; c_group < total_channels / 4; c_group++) {
        for (int w_idx = 0; w_idx < src.w; w_idx++) {
            float* dst_ptr = dst.row(c_group) + w_idx * 4;
            for (int c = 0; c < 4; c++) {
                dst_ptr[c] = src.row(c_group * 4 + c)[w_idx];
            }
        }
    }
}

static void convert_pack4_to_pack1(const Mat& src, Mat& dst)
{
    int total_channels = src.c * 4;
    dst.create(src.w, total_channels, 4);
    
    for (int c_group = 0; c_group < src.c; c_group++) {
        for (int w_idx = 0; w_idx < src.w; w_idx++) {
            const float* src_ptr = src.row(c_group) + w_idx * 4;
            for (int c = 0; c < 4; c++) {
                dst.row(c_group * 4 + c)[w_idx] = src_ptr[c];
            }
        }
    }
}

// Test case structure
struct TestCase {
    int input_width;
    int input_channels;
    int output_channels;
    int kernel_w;
    int stride_w;
    int dilation_w;
    int activation_type;
    bool use_bias;
    std::string description;
};

// Compare two matrices with tolerance
static bool compare_matrices(const Mat& a, const Mat& b, float tolerance = 1e-4, bool verbose = false)
{
    if (a.w != b.w || a.c != b.c) {
        if (verbose) {
            std::cout << "Matrix dimensions mismatch: (" << a.w << "x" << a.c 
                     << ") vs (" << b.w << "x" << b.c << ")" << std::endl;
        }
        return false;
    }
    
    float max_diff = 0.0f;
    int error_count = 0;
    int total_elements = a.w * a.c;
    
    for (int i = 0; i < total_elements; i++) {
        float diff = fabs(a.data[i] - b.data[i]);
        if (diff > tolerance) {
            error_count++;
            if (verbose && error_count <= 10) {
                std::cout << "Element [" << i << "]: " << a.data[i] 
                         << " vs " << b.data[i] << " (diff: " << diff << ")" << std::endl;
            }
        }
        max_diff = fmax(max_diff, diff);
    }
    
    if (verbose) {
        std::cout << "Max difference: " << max_diff << std::endl;
        std::cout << "Error count: " << error_count << "/" << total_elements << std::endl;
    }
    
    return error_count == 0;
}

// Initialize test data
static void init_test_data(Mat& input, Mat& weight, Mat& bias, const TestCase& test_case)
{
    // Initialize input with random values
    srand(42); // Fixed seed for reproducible results
    for (int i = 0; i < input.w * input.c; i++) {
        input.data[i] = (float(rand()) / RAND_MAX - 0.5f) * 2.0f; // [-1, 1]
    }
    
    // Initialize weights with small random values
    int weight_size = test_case.output_channels * test_case.input_channels * test_case.kernel_w;
    for (int i = 0; i < weight_size; i++) {
        weight.data[i] = (float(rand()) / RAND_MAX - 0.5f) * 0.2f; // [-0.1, 0.1]
    }
    
    // Initialize bias
    if (test_case.use_bias) {
        for (int i = 0; i < test_case.output_channels; i++) {
            bias.data[i] = (float(rand()) / RAND_MAX - 0.5f) * 0.1f; // [-0.05, 0.05]
        }
    }
}

// Run a single test case
static bool run_test_case(const TestCase& test_case, bool verbose = false)
{
    if (verbose) {
        std::cout << "\n--- Testing: " << test_case.description << " ---" << std::endl;
        std::cout << "Input: " << test_case.input_width << "x" << test_case.input_channels << std::endl;
        std::cout << "Output channels: " << test_case.output_channels << std::endl;
        std::cout << "Kernel: " << test_case.kernel_w << ", Stride: " << test_case.stride_w << std::endl;
    }
    
    // Only test 2x2 kernel with stride 2 for packed optimization
    bool can_use_packed = (test_case.kernel_w == 2 && test_case.stride_w == 2 && test_case.dilation_w == 1);
    bool can_pack_input = (test_case.input_channels % 4 == 0);
    bool can_pack_output = (test_case.output_channels % 4 == 0);
    
    if (!can_use_packed || !can_pack_output) {
        if (verbose) {
            std::cout << "Skipping packed test (not applicable for this case)" << std::endl;
        }
        return true; // Skip this test
    }
    
    // Calculate output dimensions
    int kernel_extent_w = test_case.dilation_w * (test_case.kernel_w - 1) + 1;
    int output_width = (test_case.input_width - 1) * test_case.stride_w + kernel_extent_w;
    
    // Create input data
    Mat input_pack1(test_case.input_width, test_case.input_channels);
    Mat weight_standard(test_case.output_channels * test_case.input_channels * test_case.kernel_w, 1);
    Mat bias_standard;
    if (test_case.use_bias) {
        bias_standard.create(test_case.output_channels, 1);
    }
    
    // Initialize test data
    init_test_data(input_pack1, weight_standard, bias_standard, test_case);
    
    // Create output matrices
    Mat output_standard(output_width, test_case.output_channels);
    Mat output_pack4_from_pack1(output_width, test_case.output_channels / 4, 16, 4);
    Mat output_pack4_from_pack4, input_pack4;
    
    Option opt;
    opt.use_packing_layout = true;
    
    // Test 1: Standard implementation (pack1 -> pack1)
    int ret1 = deconvolution1d_standard(input_pack1, output_standard, weight_standard, bias_standard, 
                                       test_case.kernel_w, test_case.stride_w, test_case.dilation_w, 
                                       test_case.activation_type, Mat(), opt);
    
    if (ret1 != 0) {
        if (verbose) std::cout << "Standard implementation failed!" << std::endl;
        return false;
    }
    
    // Test 2: Pack1to4 implementation (pack1 -> pack4)
    if (can_pack_output) {
        Mat weight_pack1to4, bias_pack4;
        transform_weight_for_pack4(weight_standard, weight_pack1to4, test_case.kernel_w, 
                                  test_case.input_channels, test_case.output_channels, false, true);
        transform_bias_for_pack4(bias_standard, bias_pack4, test_case.output_channels, true);
        
        int ret2 = deconv1d_k2s2_pack1to4_neon(input_pack1, output_pack4_from_pack1, weight_pack1to4, bias_pack4,
                                              test_case.activation_type, Mat(), opt);
        
        if (ret2 != 0) {
            if (verbose) std::cout << "Pack1to4 implementation failed!" << std::endl;
            return false;
        }
        
        // Convert pack4 output back to pack1 for comparison
        Mat output_from_pack1to4;
        convert_pack4_to_pack1(output_pack4_from_pack1, output_from_pack1to4);
        
        // Compare results
        bool pack1to4_match = compare_matrices(output_standard, output_from_pack1to4, 1e-4, verbose);
        if (!pack1to4_match) {
            if (verbose) std::cout << "Pack1to4 results don't match standard!" << std::endl;
            return false;
        }
        
        if (verbose) std::cout << "Pack1to4 test PASSED" << std::endl;
    }
    
    // Test 3: Pack4to4 implementation (pack4 -> pack4)
    if (can_pack_input && can_pack_output) {
        convert_pack1_to_pack4(input_pack1, input_pack4);
        output_pack4_from_pack4.create(output_width, test_case.output_channels / 4, 16, 4);
        
        Mat weight_pack4to4, bias_pack4;
        transform_weight_for_pack4(weight_standard, weight_pack4to4, test_case.kernel_w, 
                                  test_case.input_channels, test_case.output_channels, true, true);
        transform_bias_for_pack4(bias_standard, bias_pack4, test_case.output_channels, true);
        
        int ret3 = deconv1d_k2s2_pack4_neon(input_pack4, output_pack4_from_pack4, weight_pack4to4, bias_pack4,
                                           test_case.activation_type, Mat(), opt);
        
        if (ret3 != 0) {
            if (verbose) std::cout << "Pack4to4 implementation failed!" << std::endl;
            return false;
        }
        
        // Convert pack4 output back to pack1 for comparison
        Mat output_from_pack4to4;
        convert_pack4_to_pack1(output_pack4_from_pack4, output_from_pack4to4);
        
        // Compare results
        bool pack4to4_match = compare_matrices(output_standard, output_from_pack4to4, 1e-4, verbose);
        if (!pack4to4_match) {
            if (verbose) std::cout << "Pack4to4 results don't match standard!" << std::endl;
            return false;
        }
        
        if (verbose) std::cout << "Pack4to4 test PASSED" << std::endl;
    }
    
    if (verbose) std::cout << "All packed tests PASSED" << std::endl;
    return true;
}

int main()
{
    std::cout << "=== NCNN Deconvolution1D Packed Implementation Test ===" << std::endl;
    
#if __ARM_NEON
    std::cout << "ARM NEON support: ENABLED" << std::endl;
#else
    std::cout << "ARM NEON support: DISABLED" << std::endl;
#endif

    // Define test cases (focus on 2x2 kernel with stride 2 and channel counts divisible by 4)
    std::vector<TestCase> test_cases = {
        // Basic pack4 compatible cases
        {8, 4, 8, 2, 2, 1, 0, true, "2x2 kernel, 4->8 channels, no activation, with bias"},
        {8, 4, 8, 2, 2, 1, 1, true, "2x2 kernel, 4->8 channels, ReLU activation, with bias"},
        {8, 4, 8, 2, 2, 1, 2, false, "2x2 kernel, 4->8 channels, LeakyReLU activation, no bias"},
        
        // Larger channel counts
        {16, 8, 16, 2, 2, 1, 0, true, "2x2 kernel, 8->16 channels, no activation"},
        {16, 12, 16, 2, 2, 1, 1, true, "2x2 kernel, 12->16 channels, ReLU activation"},
        {32, 16, 32, 2, 2, 1, 1, true, "2x2 kernel, 16->32 channels, ReLU activation"},
        
        // Edge cases
        {4, 4, 4, 2, 2, 1, 0, false, "2x2 kernel, 4->4 channels, small input, no bias"},
        {64, 32, 64, 2, 2, 1, 2, true, "2x2 kernel, 32->64 channels, large input"},
    };
    
    bool all_tests_passed = true;
    int passed_count = 0;
    int total_count = test_cases.size();
    
    // Run all test cases
    for (const auto& test_case : test_cases) {
        bool result = run_test_case(test_case, false); // Set to true for verbose output
        if (result) {
            passed_count++;
            std::cout << "✓ " << test_case.description << std::endl;
        } else {
            all_tests_passed = false;
            std::cout << "✗ " << test_case.description << std::endl;
        }
    }
    
    // Summary
    std::cout << "\n=== Test Summary ===" << std::endl;
    std::cout << "Passed: " << passed_count << "/" << total_count << std::endl;
    
    if (all_tests_passed) {
        std::cout << "All packed tests PASSED! ✓" << std::endl;
        return 0;
    } else {
        std::cout << "Some packed tests FAILED! ✗" << std::endl;
        return 1;
    }
}
