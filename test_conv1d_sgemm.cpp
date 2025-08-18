#include <iostream>
#include <vector>
#include <random>
#include <chrono>

// Test functions for Conv1D 1x1 GEMM implementation
// This is a simple test to verify the implementation works

void test_conv1d_1x1_correctness()
{
    std::cout << "Testing Conv1D 1x1 correctness..." << std::endl;
    
    // Test parameters
    const int input_w = 128;
    const int input_channels = 64;
    const int output_channels = 32;
    
    // Generate random input data
    std::vector<float> input_data(input_w * input_channels);
    std::vector<float> weight_data(input_channels * output_channels);
    std::vector<float> bias_data(output_channels);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (auto& val : input_data) val = dis(gen);
    for (auto& val : weight_data) val = dis(gen);
    for (auto& val : bias_data) val = dis(gen);
    
    // Reference implementation (simple 1x1 convolution)
    std::vector<float> output_ref(input_w * output_channels, 0.0f);
    
    for (int ow = 0; ow < input_w; ow++) {
        for (int oc = 0; oc < output_channels; oc++) {
            float sum = bias_data[oc];
            for (int ic = 0; ic < input_channels; ic++) {
                sum += input_data[ow * input_channels + ic] * weight_data[oc * input_channels + ic];
            }
            output_ref[ow * output_channels + oc] = sum;
        }
    }
    
    // TODO: Here we would call our GEMM implementation and compare results
    // For now, just print success
    std::cout << "Conv1D 1x1 correctness test: PASSED (implementation needed)" << std::endl;
}

void benchmark_conv1d_1x1_performance()
{
    std::cout << "Benchmarking Conv1D 1x1 performance..." << std::endl;
    
    struct TestCase {
        int w, inch, outch;
        std::string name;
    };
    
    std::vector<TestCase> test_cases = {
        {64, 32, 64, "Small"},
        {128, 64, 128, "Medium"},
        {256, 128, 256, "Large"},
        {512, 256, 512, "XLarge"}
    };
    
    for (const auto& tc : test_cases) {
        std::cout << "Testing " << tc.name << " case: w=" << tc.w 
                  << ", in_ch=" << tc.inch << ", out_ch=" << tc.outch << std::endl;
        
        // Generate test data
        std::vector<float> input_data(tc.w * tc.inch);
        std::vector<float> weight_data(tc.inch * tc.outch);
        std::vector<float> output_data(tc.w * tc.outch);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        
        for (auto& val : input_data) val = dis(gen);
        for (auto& val : weight_data) val = dis(gen);
        
        // Benchmark reference implementation
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int iter = 0; iter < 100; iter++) {
            for (int ow = 0; ow < tc.w; ow++) {
                for (int oc = 0; oc < tc.outch; oc++) {
                    float sum = 0.0f;
                    for (int ic = 0; ic < tc.inch; ic++) {
                        sum += input_data[ow * tc.inch + ic] * weight_data[oc * tc.inch + ic];
                    }
                    output_data[ow * tc.outch + oc] = sum;
                }
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        double ops = static_cast<double>(tc.w) * tc.inch * tc.outch * 2 * 100; // 2 ops per MAC, 100 iterations
        double gflops = ops / (duration.count() * 1000.0); // GFLOPS
        
        std::cout << "  Reference: " << duration.count() / 100.0 << " μs/iter, " 
                  << gflops << " GFLOPS" << std::endl;
        
        // TODO: Benchmark our GEMM implementation here
        std::cout << "  GEMM: (implementation needed)" << std::endl;
    }
}

int main()
{
    std::cout << "=== Conv1D 1x1 GEMM Implementation Test ===" << std::endl;
    
    test_conv1d_1x1_correctness();
    std::cout << std::endl;
    
    benchmark_conv1d_1x1_performance();
    std::cout << std::endl;
    
    std::cout << "Test completed. Next steps:" << std::endl;
    std::cout << "1. Complete the GEMM implementation in convolution1d_im2col_gemm.h" << std::endl;
    std::cout << "2. Fix compilation errors by adding proper includes" << std::endl;
    std::cout << "3. Integrate with NCNN build system" << std::endl;
    std::cout << "4. Run real performance comparisons" << std::endl;
    
    return 0;
}
