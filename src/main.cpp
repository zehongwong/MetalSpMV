//
// Created by Zehong Wang on 2023-04-25.
//

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "spmv_calculator.h"

int main(int argc, char **argv) {

    // Initialization
    char *file_location   = argv[1];
    if (file_location == nullptr) {
        Logger::error("Please specify a file location. Example: ./spmv data/af_shell10.mtx");
        Logger::error("Download link: https://suitesparse-collection-website.herokuapp.com/MM/Schenk_AFE/af_shell10.tar.gz");
        Logger::error("Extract and put the .mtx file in the data folder");
        exit(-1);
    }
    SpmvCalculator spmv_calculator(file_location);

    // Do SPMV
    spmv_calculator.do_cpu_serial(); // CPU Serial
    spmv_calculator.do_cpu_omp();    // CPU OpenMP
    spmv_calculator.do_gpu_csr();    // GPU CSR
    spmv_calculator.do_gpu_csr_loop_unrolling(); // GPU CSR Loop Unrolling
    spmv_calculator.do_gpu_coo();    // GPU COO

    // Verify
    spmv_calculator.verify();

    return 0;
}