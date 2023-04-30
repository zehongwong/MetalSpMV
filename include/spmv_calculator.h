//
// Created by Zehong Wang on 2023-04-23.
//

#ifndef METALSPMV_SPMVCALCULATOR_H
#define METALSPMV_SPMVCALCULATOR_H

#define REPEAT_TIMES 100

#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"
#include "logger.h"
#include <map>

enum class KernelFunc {
    CSR_BASIC = 2,
    CSR_LOOP_UNROLLING,
    COO,
};
extern std::map<std::string, KernelFunc> kernel_func_map;
extern std::map<KernelFunc, std::string> kernel_func_name_map;

class SpmvCalculator {
public:
    SpmvCalculator() = default; // Default constructor
    SpmvCalculator(const std::string& file_name);  // Load Matrix-Vector and Metal Library
    ~SpmvCalculator(); // Release source

    void do_gpu_csr();                 // GPU with CSR format
    void do_gpu_csr_loop_unrolling();  // GPU with CSR format, Optimized with loop unrolling
    void do_gpu_coo();                 // GPU with COO format
    void do_cpu_serial();              // CPU with serial
    void do_cpu_omp();                 // CPU with OpenMP

    void verify();                     // Verify the result

private:

    // Input Matrix and Vector
    size_t matrix_size; // Matrix size
    size_t nnz;         // Number of non-zero elements
    MTL::Buffer *csr_row_ptr; // CSR Row pointer
    MTL::Buffer *csr_col_ind; // CSR Column index
    MTL::Buffer *csr_data;    // CSR Non-zero values
    MTL::Buffer *coo_row_ind; // COO Row index
    MTL::Buffer *coo_col_ind; // COO Column index
    MTL::Buffer *coo_data;    // COO Non-zero values
    MTL::Buffer *x;           // Input Vector
    MTL::Buffer *y;           // Output Vector

    // Output Vector
    float* y_cpu_serial;
    float* y_cpu_omp;
    float* y_gpu_csr;
    float* y_gpu_csr_loop_unrolling;
    float* y_gpu_coo;

    // Metal Device, Function Pipeline State Object, Command Queue
    MTL::Device* device{};
    std::map<KernelFunc, MTL::ComputePipelineState*> spmv_ps_map;
    MTL::CommandQueue* commandQueue{};

    void load_matrix_vector(const std::string& filename); // Load the matrix from file and mock the input vector
    void load_metal_setup();                              // Load Metal Library and Function Pipeline State Object
    MTL::CommandBuffer* pack_metal_command(KernelFunc func);   // Build Metal Command Encoder for CSR
    bool compare(float* a, float* b, float eps) const; // Compare two float arrays
};

#endif // METALSPMV_SPMVCALCULATOR_H
