//
// Created by Zehong Wang on 2023-04-25.
//

#include "spmv_calculator.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <omp.h>
#include <gsl/gsl_spmatrix.h>

std::map<std::string, KernelFunc> kernel_func_map = {
        {"spmv_csr", KernelFunc::CSR_BASIC},
        {"spmv_csr_loop_unrolling", KernelFunc::CSR_LOOP_UNROLLING},
        {"spmv_coo", KernelFunc::COO},
};
std::map<KernelFunc, std::string> kernel_func_name_map = {
        {KernelFunc::CSR_BASIC, "spmv_csr"},
        {KernelFunc::CSR_LOOP_UNROLLING, "spmv_csr_loop_unrolling"},
        {KernelFunc::COO, "spmv_coo"},
};

// Constructor and destructor
SpmvCalculator::SpmvCalculator(const std::string &filename) {

    // Load Metal Setup
    load_metal_setup();

    // Load Matrix and Vector
    Logger::info("Loading matrix and vector...");
    auto start = std::chrono::high_resolution_clock::now();
    load_matrix_vector(filename);
    Logger::time("Loaded matrix and vector", start);
}

SpmvCalculator::~SpmvCalculator() = default;

void SpmvCalculator::load_metal_setup() {
    // Create the metal device
    device = MTL::CreateSystemDefaultDevice();

    // Load Metal Library for SpMV
    NS::Error *error;
    NS::String *filePath =
            NS::String::string("spmv.metallib", NS::ASCIIStringEncoding);
    auto lib = device->newLibrary(filePath, &error);
    if (!lib) {
        Logger::error("Failed to load Metal Library for SpMV. Try 'make kernel' to generate the library.");
        std::exit(-1);
    }

    Logger::info("Loaded Metal Library for SpMV.");

    // Load Metal Kernel Function for SpMV
    auto fnNames = lib->functionNames();
    for (size_t i = 0; i < fnNames->count(); i++) {
        auto desc = fnNames->object(i)->description();
        auto func = desc->utf8String();
        if (kernel_func_map.find(func) == kernel_func_map.end()) {
            Logger::warn("Unknown Metal Kernel Function: " + std::string(func));
            continue;
        }
        spmv_ps_map[kernel_func_map[func]] = device->newComputePipelineState(
                lib->newFunction(desc), &error);
        if (!spmv_ps_map[kernel_func_map[func]]) {
            Logger::error("Failed to load Metal Kernel Function: " + std::string(func));
            std::exit(-1);
        } else {
            Logger::info("Loaded Metal Kernel Function: " + std::string(func));
        }
    }

    // Initialize the command queue
    commandQueue = device->newCommandQueue();
}

// Get the matrix from file and mock the input vector
void SpmvCalculator::load_matrix_vector(const std::string& filename) {

    // Read the matrix from file
    FILE *f;
    f = fopen(filename.c_str(), "r");
    if (f == nullptr) {
        Logger::error("Failed to open file: " + filename + ". Please check the file path.");
        Logger::error("Download link: https://suitesparse-collection-website.herokuapp.com/MM/Schenk_AFE/af_shell10.tar.gz");
        Logger::error("Extract and put the .mtx file in the data folder");
        std::exit(-1);
    }
    gsl_spmatrix* spm_coo = gsl_spmatrix_fscanf(f);
    fclose(f);

    // Get the matrix size and nnz
    matrix_size = spm_coo->size1;
    nnz = spm_coo->nz;

    // Malloc Metal Buffers for CSR/COO
    coo_row_ind = device->newBuffer(sizeof(int)*nnz, MTL::ResourceStorageModeShared);
    coo_col_ind = device->newBuffer(sizeof(int)*nnz, MTL::ResourceStorageModeShared);
    coo_data = device->newBuffer(sizeof(float)*nnz, MTL::ResourceStorageModeShared);
    csr_row_ptr = device->newBuffer(sizeof(int)*(matrix_size+1), MTL::ResourceStorageModeShared);
    csr_col_ind = device->newBuffer(sizeof(int)*nnz, MTL::ResourceStorageModeShared);
    csr_data = device->newBuffer(sizeof(float)*nnz, MTL::ResourceStorageModeShared);
    x = device->newBuffer(sizeof(float)*matrix_size, MTL::ResourceStorageModeShared);
    y = device->newBuffer(sizeof(float)*matrix_size, MTL::ResourceStorageModeShared);
    auto coo_row_ind_cpp = (int*)coo_row_ind->contents();
    auto coo_col_ind_cpp = (int*)coo_col_ind->contents();
    auto coo_data_cpp = (float*)coo_data->contents();
    auto csr_row_ptr_cpp = (int*)csr_row_ptr->contents();
    auto csr_col_ind_cpp = (int*)csr_col_ind->contents();
    auto csr_data_cpp = (float*)csr_data->contents();
    auto x_cpp = (float*)x->contents();

    // Fill the COO buffers
    for (int i = 0; i < nnz; i++) {
        coo_row_ind_cpp[i] = spm_coo->i[i];
        coo_col_ind_cpp[i] = spm_coo->p[i];
        coo_data_cpp[i] = (float)spm_coo->data[i];
    }

    // Convert to CSR format
    gsl_spmatrix* spm_csr = gsl_spmatrix_crs(spm_coo);
    gsl_spmatrix_free(spm_coo);

    // Fill the CSR buffers
    for (int i = 0; i < matrix_size+1; i++) {
        csr_row_ptr_cpp[i] = spm_csr->p[i];
    }
    for (int i = 0; i < nnz; i++) {
        csr_col_ind_cpp[i] = spm_csr->i[i];
        csr_data_cpp[i] = (float)spm_csr->data[i];
    }
    gsl_spmatrix_free(spm_csr);

    // Mock the input vector
    std::fill(x_cpp, x_cpp + matrix_size, 1.0f);
}

MTL::CommandBuffer* SpmvCalculator::pack_metal_command(KernelFunc func) {
    // Create buffers
    MTL::CommandBuffer *command_buffer = commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder *compute_encoder = command_buffer->computeCommandEncoder();

    // Encode the compute command
    auto spmv_function_PSO = spmv_ps_map[func];
    if (!spmv_function_PSO) {
        Logger::error("Failed to load Metal Kernel Function: " + kernel_func_name_map[func]);
        std::exit(-1);
    }
    compute_encoder->setComputePipelineState(spmv_function_PSO);

    // Set the data buffer, total threads, thread group size
    MTL::Size grid_size{};
    MTL::Size thread_group_size{};
    if (func == KernelFunc::COO) { // COO Buffer
        compute_encoder->setBuffer(coo_row_ind, 0, 0);
        compute_encoder->setBuffer(coo_col_ind, 0, 1);
        compute_encoder->setBuffer(coo_data, 0, 2);
        grid_size = MTL::Size(nnz, 1, 1);
        NS::UInteger _thread_group_size = std::min(spmv_function_PSO->maxTotalThreadsPerThreadgroup(), nnz);
        thread_group_size = MTL::Size(_thread_group_size, 1, 1);
    } else { // CSR Buffer
        compute_encoder->setBuffer(csr_row_ptr, 0, 0);
        compute_encoder->setBuffer(csr_col_ind, 0, 1);
        compute_encoder->setBuffer(csr_data, 0, 2);
        grid_size = MTL::Size(matrix_size, 1, 1);
        NS::UInteger _thread_group_size = std::min(spmv_function_PSO->maxTotalThreadsPerThreadgroup(), matrix_size);
        thread_group_size = MTL::Size(_thread_group_size, 1, 1);
    }
    compute_encoder->setBuffer(x, 0, 3);
    compute_encoder->setBuffer(y, 0, 4);

    // Dispatch the threads
    compute_encoder->dispatchThreads(grid_size, thread_group_size);
    compute_encoder->endEncoding();
    return command_buffer;
}

void SpmvCalculator::do_gpu_csr() {
    y_gpu_csr = new float[matrix_size];
    Logger::info("Running SpMV on GPU CSR Version...");
    auto start = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < REPEAT_TIMES; ++ k) {
        // Create buffers
        MTL::CommandBuffer *command_buffer = pack_metal_command(KernelFunc::CSR_BASIC);
        // Run
        command_buffer->commit();
        command_buffer->waitUntilCompleted();
    }
    Logger::time("Finished SpMV on GPU CSR Version", start);
    memcpy(y_gpu_csr, (float *)y->contents(), matrix_size * sizeof(float));
}

void SpmvCalculator::do_gpu_coo() {
    y_gpu_coo = new float[matrix_size];
    Logger::info("Running SpMV on GPU COO Version...");
    auto start = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < REPEAT_TIMES; ++ k) {
        memset(y->contents(), 0, matrix_size * sizeof(float));
        // Create buffers
        MTL::CommandBuffer *command_buffer = pack_metal_command(KernelFunc::COO);
        // Run
        command_buffer->commit();
        command_buffer->waitUntilCompleted();
    }
    Logger::time("Finished SpMV on GPU COO Version", start);
    memcpy(y_gpu_coo, (float *)y->contents(), matrix_size * sizeof(float));
}

void SpmvCalculator::do_cpu_serial() {
    // Get Data
    auto row_ptr = (int*)csr_row_ptr->contents();
    auto col_ind = (int*)csr_col_ind->contents();
    auto data = (float*)csr_data->contents();
    auto input_vector = (float*)x->contents();

    // Run
    y_cpu_serial = new float[matrix_size];
    Logger::info("Running SpMV on CPU Serial Version...");
    auto start = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < REPEAT_TIMES; ++ k) {
        for (int i = 0; i < matrix_size; ++ i) {
            float sum = 0.0f;
            for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++ j) {
                sum += data[j] * input_vector[col_ind[j]];
            }
            y_cpu_serial[i] = sum;
        }
    }
    Logger::time("Finished SpMV on CPU Serial Version", start);
}

void SpmvCalculator::do_gpu_csr_loop_unrolling() {
    y_gpu_csr_loop_unrolling = new float[matrix_size];
    Logger::info("Running SpMV on GPU CSR Version optimized with Loop Unrolling...");
    auto start = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < REPEAT_TIMES; ++ k) {
        // Create buffers
        MTL::CommandBuffer *command_buffer = pack_metal_command(KernelFunc::CSR_LOOP_UNROLLING);
        // Run
        command_buffer->commit();
        command_buffer->waitUntilCompleted();
    }
    Logger::time("Finished SpMV on GPU CSR Version optimized with Loop Unrolling", start);
    memcpy(y_gpu_csr_loop_unrolling, (float *)y->contents(), matrix_size * sizeof(float));
}

void SpmvCalculator::do_cpu_omp() {
    // Get Data
    auto row_ptr = (int*)csr_row_ptr->contents();
    auto col_ind = (int*)csr_col_ind->contents();
    auto data = (float*)csr_data->contents();
    auto input_vector = (float*)x->contents();

    // Run
    y_cpu_omp = new float[matrix_size];
    Logger::info("Running SpMV on CPU OpenMP Version...");
    auto start = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < REPEAT_TIMES; ++ k) {
#pragma omp parallel for
        for (int i = 0; i < matrix_size; ++ i) {
            float sum = 0.0f;
            for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++ j) {
                sum += data[j] * input_vector[col_ind[j]];
            }
            y_cpu_omp[i] = sum;
        }
    }
    Logger::time("Finished SpMV on CPU OpenMP Version", start);
}

void SpmvCalculator::verify() {
    Logger::info("Running Verification...");

    // Verification Database
    if (y_cpu_serial == nullptr) {
        Logger::error("CPU Serial Version is not calculated!");
        std::exit(-1);
    }

    // CPU OpenMP Version
    if (y_cpu_omp == nullptr) {
        Logger::warn("CPU OpenMP Version is not calculated!");
    } else if (compare(y_cpu_serial, y_cpu_omp, 1e-5)) {
        Logger::info("CPU OpenMP Version verification passed!");
    } else {
        Logger::error("CPU OpenMP Version verification failed!");
    }

    // GPU CSR Version
    if (y_gpu_csr == nullptr) {
        Logger::warn("GPU CSR Version is not calculated!");
    } else if (compare(y_cpu_serial, y_gpu_csr, 1e-5)) {
        Logger::info("GPU CSR Version verification passed!");
    } else {
        Logger::error("GPU CSR Version verification failed!");
    }

    // GPU CSR Version optimized with Loop Unrolling
    if (y_gpu_csr_loop_unrolling == nullptr) {
        Logger::warn("GPU CSR Version optimized with Loop Unrolling is not calculated!");
    } else if (compare(y_cpu_serial, y_gpu_csr_loop_unrolling, 1e-5)) {
        Logger::info("GPU CSR Version optimized with Loop Unrolling verification passed!");
    } else {
        Logger::error("GPU CSR Version optimized with Loop Unrolling verification failed!");
    }

    // GPU COO Version
    if (y_gpu_coo == nullptr) {
        Logger::warn("GPU COO Version is not calculated!");
    } else if (compare(y_cpu_serial, y_gpu_coo, 1)) { // COO precision is lower
        Logger::info("GPU COO Version verification passed!");
    } else {
        Logger::error("GPU COO Version verification failed!");
    }

    Logger::info("Finished Verification!");
}

bool SpmvCalculator::compare(float* src, float* dst, float eps) const {
    // Verify the result
    for (size_t i = 0; i < matrix_size; ++i) {
        if (std::abs(src[i] - dst[i]) > eps) {
            return false;
        }
    }
    return true;
}