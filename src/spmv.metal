#include <metal_stdlib>
using namespace metal;

kernel void spmv_csr(device const int* row_ptr,
                     device const int* col_index,
                     device const float* val,
                     device const float* x,
                     device float* result,
                     uint index [[thread_position_in_grid]]
) {
    int start = row_ptr[index];
    int end = row_ptr[index + 1];
    float sum = 0.0f;
    for (int i = start; i < end; ++i) {
        sum += val[i] * x[col_index[i]];
    }
    result[index] = sum;
}

kernel void spmv_csr_loop_unrolling(device const int* row_ptr,
                                    device const int* col_index,
                                    device const float* val,
                                    device const float* x,
                                    device float* result,
                                    constant ushort* max_simd_width,
                                    uint index [[thread_position_in_grid]]
) {
    int start = row_ptr[index];
    int end = row_ptr[index + 1];
    float sum = 0.0f;
    int i = start;
    for (; i + 3 < end; i += 4) {
        sum += val[i] * x[col_index[i]];
        sum += val[i + 1] * x[col_index[i + 1]];
        sum += val[i + 2] * x[col_index[i + 2]];
        sum += val[i + 3] * x[col_index[i + 3]];
    }
    for (; i < end; ++i) {
        sum += val[i] * x[col_index[i]];
    }
    result[index] = sum;
}

// Metal kernel function for SpMV with COO format
kernel void spmv_coo(
        device const int* row_index,
        device const int* col_index,
        device const float* val,
        device const float* x,
        device atomic_float* result,
        uint index [[thread_position_in_grid]]
){
    // Using atomic_fetch_add_explicit on a per-element basis for the result vector
    atomic_fetch_add_explicit(
            &result[row_index[index]], val[index] * x[col_index[index]], memory_order_relaxed);
}