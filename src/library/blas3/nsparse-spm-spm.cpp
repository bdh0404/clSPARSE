/**
 * @brief General Sparse Matrix-Matrix Multiplication Program
 * 
 */
/* ************************************************************************
* The MIT License (MIT)
* Copyright 2014-2015 University of Copenhagen
*  Permission is hereby granted, free of charge, to any person obtaining a copy
*  of this software and associated documentation files (the "Software"), to deal
*  in the Software without restriction, including without limitation the rights
*  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
*  copies of the Software, and to permit persons to whom the Software is
*  furnished to do so, subject to the following conditions:

*  The above copyright notice and this permission notice shall be included in
*  all copies or substantial portions of the Software.

*  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
*  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
*  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
*  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
*  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
*  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
*  THE SOFTWARE.
* ************************************************************************ */

/* ************************************************************************
*  < A CUDA/OpenCL General Sparse Matrix-Matrix Multiplication Program >
*
*  < See papers:
*  1. Weifeng Liu and Brian Vinter, "A Framework for General Sparse
*      Matrix-Matrix Multiplication on GPUs and Heterogeneous
*      Processors," Journal of Parallel and Distributed Computing, 2015.
*  2. Weifeng Liu and Brian Vinter, "An Efficient GPU General Sparse
*      Matrix-Matrix Multiplication for Irregular Data," Parallel and
*      Distributed Processing Symposium, 2014 IEEE 28th International
*      (IPDPS '14), pp.370-381, 19-23 May 2014.
*  for details. >
* ************************************************************************ */

#include "include/clSPARSE-private.hpp"
#include "internal/clsparse-control.hpp"
#include "internal/kernel-cache.hpp"
#include "internal/kernel-wrap.hpp"
#include "internal/clsparse-internal.hpp"
#include "clSPARSE-error.h"
#include <iostream>

#include <cmath>

// maximum 256 threads per workgroup
#define GROUPSIZE_256 256
// the number of bins of the inner-product numbers
#define NIP_SEGMENTS 16
// the number of bins of the non-zero numbers
#define NNZ_SEGMENTS 15
//#define WARPSIZE_NV_2HEAP 64
#define value_type float
#define index_type int
#define NSPARSE_SUCCESS 0
// maximum integer numbers that can be in local memory
#define MAX_HASH_SIZE 8192

using namespace std;

/**
 * @brief Compute the number of products of each C's row
 * 
 * @param _m Number of C's rows
 * @param csrRowPtrA A's row pointer array
 * @param csrColIndA A's column indices array
 * @param csrRowPtrB B's row pointer array
 * @param csrRowCInnProdNum Buffer for storing the result
 * @param clMaxInnProd Buffer for storing the maximum number of products of all C's rows
 * @param control Control for Library
 * @return clsparseStatus If CL_SUCCESS, success, if not, fails.
 */
clsparseStatus compute_CInnProdNum(int _m, cl_mem csrRowPtrA, cl_mem csrColIndA, cl_mem csrRowPtrB,
                                      cl_mem csrRowCInnProdNum, cl_mem clMaxInnProd, 
                                      clsparseControl control)
{
    const std::string params = std::string() + "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type 
        + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type;

    cl::Kernel kernel = KernelCache::get(control->queue, "SpGEMM_compute_InnProdNum_kernel",
                                         "compute_InnProdNum_kernel", params);

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    int num_threads = GROUPSIZE_256;
    size_t num_blocks = ceil((double)_m / (double)num_threads);

    szLocalWorkSize[0] = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    KernelWrap kWrapper(kernel);

    kWrapper << csrRowPtrA << csrColIndA << csrRowPtrB << csrRowCInnProdNum
             << cl::Local(256 * sizeof(int)) << clMaxInnProd << _m;

    cl::NDRange local(szLocalWorkSize[0]);
    cl::NDRange global(szGlobalWorkSize[0]);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        CLSPARSE_V(status, "compute_CInnProdNum");
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

/**
 * @brief Count the number of rows in each intermediate products' bin
 * 
 * @param _m Number of C's rows
 * @param csrRowCInnProdNum Number of products in each C's row
 * @param innBin Bins of C's rows
 * @param control Control for Library
 * @return clsparseStatus If CL_SUCCESS, success, if not, fails.
 */
clsparseStatus compute_nipBin(int _m, cl_mem csrRowCInnProdNum, cl_mem innBin, clsparseControl control)
{
    const std::string params = std::string() + "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type 
        + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type;

    cl::Kernel kernel = KernelCache::get(control->queue, "SpGEMM_computeNipBin_kernels",
                                         "compute_nipBin_kernel", params);

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    int num_threads = GROUPSIZE_256;
    size_t num_blocks = ceil((double)_m / (double)num_threads);

    szLocalWorkSize[0] = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    KernelWrap kWrapper(kernel);

    kWrapper << csrRowCInnProdNum << cl::Local(NIP_SEGMENTS * sizeof(int)) << innBin << _m;

    cl::NDRange local(szLocalWorkSize[0]);
    cl::NDRange global(szGlobalWorkSize[0]);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        CLSPARSE_V(status, "compute_nipBin");
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

/**
 * @brief Reorder C's rows based on bins
 * 
 * @param _m Number of C's rows
 * @param csrRowCInnProdNum Number of products in each C's row
 * @param clinnPtr Base pointers for each bin
 * @param csrRowCReorder Buffer for storing reordered C's rows' numbers
 * @param control Control for Library
 * @return clsparseStatus If CL_SUCCESS, success, if not, fails.
 */
clsparseStatus compute_reorderRowNip(int _m, cl_mem csrRowCInnProdNum, cl_mem clinnPtr,
                                     cl_mem csrRowCReorder, clsparseControl control)
{
    const std::string params = std::string() + "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type 
        + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type;

    cl::Kernel kernel = KernelCache::get(control->queue, "SpGEMM_computeReorderRowNip_kernels",
                                         "compute_ReorderRowNip_kernel", params);

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    int num_threads = GROUPSIZE_256;
    size_t num_blocks = ceil((double)_m / (double)num_threads);

    szLocalWorkSize[0] = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    KernelWrap kWrapper(kernel);

    kWrapper << csrRowCInnProdNum << clinnPtr << csrRowCReorder << _m;

    cl::NDRange local(szLocalWorkSize[0]);
    cl::NDRange global(szGlobalWorkSize[0]);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        CLSPARSE_V(status, "compute_reorderRowNip");
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

/**
 * @brief Compute the number of nonzeros of C's rows with 0 intermediate products (to 0)
 * 
 * @param csrRowCReorder Reordered C's rows
 * @param csrRowCNnzSize Buffer for storing the number of non-zeros for each C's row (to 0)
 * @param bin Number of C's rows with 0 intermediate products
 * @param control Control for Library
 * @return clsparseStatus If CL_SUCCESS, success, if not, fails.
 */
clsparseStatus compute_nnzC_0(cl_mem csrRowCReorder, cl_mem csrRowCNnzSize, int bin,
                              clsparseControl control)
{
    const std::string params = std::string() + "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type 
        + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type;

    cl::Kernel kernel = KernelCache::get(control->queue, "SpGEMM_computeNnz_kernels", "compute_nnzC_0",
                                         params);

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    int num_threads = GROUPSIZE_256;
    size_t num_blocks = ceil((double)bin / (double)num_threads);

    szLocalWorkSize[0] = GROUPSIZE_256;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    KernelWrap kWrapper(kernel);
    kWrapper << csrRowCReorder << csrRowCNnzSize << bin;

    cl::NDRange local(szLocalWorkSize[0]);
    cl::NDRange global(szGlobalWorkSize[0]);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        CLSPARSE_V(status, "compute_nnzC_0");
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

/**
 * @brief Compute the number of nonzeros of C's rows with 1 intermediate product (to 0)
 * 
 * @param csrRowCReorder Reordered C's rows
 * @param csrRowCNnzSize Buffer for storing the number of non-zeros for each C's row (to 1)
 * @param bin Number of C's rows with 1 intermediate product
 * @param ptr Baseline pointer for reading C's rows in csrRowCReorder
 * @param control Control for Library
 * @return clsparseStatus If CL_SUCCESS, success, if not, fails.
 */
clsparseStatus compute_nnzC_1(cl_mem csrRowCReorder, cl_mem csrRowCNnzSize, int bin, int ptr,
                              clsparseControl control)
{
    const std::string params = std::string() + "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type +
                               " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type;

    cl::Kernel kernel = KernelCache::get(control->queue, "SpGEMM_computeNnz_kernels", "compute_nnzC_1",
                                         params);

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    int num_threads = GROUPSIZE_256;
    size_t num_blocks = ceil((double)bin / (double)num_threads);

    szLocalWorkSize[0] = GROUPSIZE_256;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    KernelWrap kWrapper(kernel);
    kWrapper << csrRowCReorder << csrRowCNnzSize << bin << ptr;

    cl::NDRange local(szLocalWorkSize[0]);
    cl::NDRange global(szGlobalWorkSize[0]);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        CLSPARSE_V(status, "compute_nnzC_1");
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

/**
 * @brief Compute the number of non-zeros of C's rows with 2~64 intermediate products
 * 
 * @param csrRowPtrA A's row pointer array
 * @param csrColIndA A's column indices array
 * @param csrRowPtrB B's row pointer array
 * @param csrColIndB B's column indices array
 * @param csrRowCReorder Reordered C's rows
 * @param csrRowCNnzSize Buffer for storing the number of non-zeros for each C's row
 * @param bin Number of C's rows in each bin
 * @param ptr Baseline pointer of each bin for reading C's rows
 * @param innSize Maximum number of intermediate numbers of C's rows in each bin
 * @param threads_per_row number of threads that will be allocated per A's column index
 * @param control Control for Library
 * @return clsparseStatus If CL_SUCCESS, success, if not, fails.
 */
clsparseStatus compute_nnzC_pwarp(cl_mem csrRowPtrA, cl_mem csrColIndA, cl_mem csrRowPtrB,
                                  cl_mem csrColIndB, cl_mem csrRowCReorder, cl_mem csrRowCNnzSize, 
                                  int bin, int ptr, int innSize, int threads_per_row,
                                  clsparseControl control)
{
    const std::string params = std::string() + "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type 
        + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type;

    cl::Kernel kernel = KernelCache::get(control->queue, "SpGEMM_computeNnz_kernels", 
        "compute_nnzC_pwarp", params);

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    int num_threads = 256;
    size_t num_blocks = ceil((double)bin / (double)(GROUPSIZE_256 / threads_per_row));

    szLocalWorkSize[0] = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    KernelWrap kWrapper(kernel);
    kWrapper << csrRowPtrA << csrColIndA << csrRowPtrB << csrColIndB << csrRowCReorder << csrRowCNnzSize 
        << cl::Local(innSize * (GROUPSIZE_256 / threads_per_row) * sizeof(int)) << bin << ptr << innSize 
        << threads_per_row;

    cl::NDRange local(szLocalWorkSize[0]);
    cl::NDRange global(szGlobalWorkSize[0]);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        CLSPARSE_V(status, "compute_nnzC_pwarp");
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

/**
 * @brief Compute the number of non-zeros of C's rows with 65~8192 intermediate products
 * 
 * @param num_threads Number of threads for each workgroup
 * @param csrRowPtrA A's row pointer array
 * @param csrColIndA A's column indices array
 * @param csrRowPtrB B's row pointer array
 * @param csrColIndB B's column indices array
 * @param csrRowCReorder Reordered C's rows
 * @param csrRowCNnzSize Buffer for storing the number of non-zeros for each C's row
 * @param bin Number of C's rows in each bin
 * @param ptr Baseline pointer of each bin for reading C's rows
 * @param innSize Maximum number of intermediate numbers of C's rows in each bin
 * @param control Control for Library
 * @return clsparseStatus If CL_SUCCESS, success, if not, fails.
 */
clsparseStatus compute_nnzC_tb(int num_threads, cl_mem csrRowPtrA, cl_mem csrColIndA, cl_mem csrRowPtrB,
                               cl_mem csrColIndB, cl_mem csrRowCReorder, cl_mem csrRowCNnzSize, int bin, 
                               int ptr, int innSize, int threads_per_row, clsparseControl control)
{
    const std::string params = std::string() + "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type 
        + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type;

    cl::Kernel kernel = KernelCache::get(control->queue, "SpGEMM_computeNnz_kernels", "compute_nnzC_tb", 
        params);

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    szLocalWorkSize[0] = num_threads;
    szGlobalWorkSize[0] = bin;

    KernelWrap kWrapper(kernel);
    kWrapper << csrRowPtrA << csrColIndA << csrRowPtrB << csrColIndB << csrRowCReorder << csrRowCNnzSize 
        << cl::Local(2 * innSize * sizeof(int)) << bin << ptr << innSize << threads_per_row;

    cl::NDRange local(szLocalWorkSize[0]);
    cl::NDRange global(szGlobalWorkSize[0]);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        CLSPARSE_V(status, "compute_nnzC_tb");
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

/**
 * @brief Try to compute the number of non-zeros of C's rows with 8193~ intermediate products in local 
 *        memory
 * 
 * @param csrRowPtrA A's row pointer array
 * @param csrColIndA A's column indices array
 * @param csrRowPtrB B's row pointer array
 * @param csrColIndB B's column indices array
 * @param csrRowCReorder Reordered C's rows
 * @param csrRowCNnzSize Buffer for storing the number of non-zeros for each C's row
 * @param fail_count Number of C's rows that failed to count the number of non-zeros
 * @param fail_perm Buffer for storing C's rows that failed to count the number of non-zeros
 * @param bin Number of C's rows in the last bin
 * @param ptr Baseline pointer of the last bin for reading C's rows
 * @param control Control for library
 * @return clsparseStatus If CL_SUCCESS, success, if not, fails.
 */
clsparseStatus compute_nnzC_tb_large(cl_mem csrRowPtrA, cl_mem csrColIndA, cl_mem csrRowPtrB, 
                                     cl_mem csrColIndB, cl_mem csrRowCReorder, cl_mem csrRowCNnzSize, 
                                     cl_mem fail_count, cl_mem fail_perm, int bin, int ptr, 
                                     clsparseControl control)
{
    const std::string params = std::string() + "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type 
        + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type;

    cl::Kernel kernel = KernelCache::get(control->queue, "SpGEMM_computeNnz_kernels", 
        "compute_nnzC_tb_large", params);

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    int num_threads = GROUPSIZE_256;
    size_t num_blocks = ceil((double)bin / (double)num_threads);

    szLocalWorkSize[0] = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    KernelWrap kWrapper(kernel);
    kWrapper << csrRowPtrA << csrColIndA << csrRowPtrB << csrColIndB << csrRowCReorder << csrRowCNnzSize 
        << fail_count << fail_perm << cl::Local((MAX_HASH_SIZE - 1) * sizeof(int)) 
        << cl::Local(sizeof(int)) << bin << ptr;

    cl::NDRange local(szLocalWorkSize[0]);
    cl::NDRange global(szGlobalWorkSize[0]);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        CLSPARSE_V(status, "compute_nnzC_tb_large");
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

/**
 * @brief Compute the number of non-zeros of C's rows with 8193~ intermediate products in global memory 
 *        that was failed on local memory
 * 
 * @param csrRowPtrA A's row pointer array
 * @param csrColIndA A's column indices array
 * @param csrRowPtrB B's row pointer array
 * @param csrColIndB B's column indices array
 * @param fail_perm C's rows that failed to count the number of non-zeros in local memory
 * @param csrRowCNnzSize Buffer for storing the number of non-zeros for each C's row
 * @param clhashtable Global memory buffer that used for counting the number of non-zeros
 * @param fail_count The number of C's rows that failed to count the number of non-zeros in local memory
 * @param max_intprod The number of maximum intermediate products of all C's rows, size of clhashtable
 * @param clMaxNnz The number of maximum non-zero elements of all C's rows
 * @param control Control for library
 * @return clsparseStatus If CL_SUCCESS, success, if not, fails.
 */
clsparseStatus compute_nnzC_tb_global(cl_mem csrRowPtrA, cl_mem csrColIndA, cl_mem csrRowPtrB, 
                                      cl_mem csrColIndB, cl_mem fail_perm, cl_mem csrRowCNnzSize, 
                                      cl_mem clhashtable, int fail_count, int max_intprod, 
                                      cl_mem clMaxNnz, clsparseControl control)
{
    const std::string params = std::string() + "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type 
        + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type;

    cl::Kernel kernel = KernelCache::get(control->queue, "SpGEMM_computeNnz_kernels", 
        "compute_nnzC_tb_large", params);

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    int num_threads = GROUPSIZE_256;
    size_t num_blocks = ceil((double)fail_count / (double)num_threads);

    szLocalWorkSize[0] = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    KernelWrap kWrapper(kernel);
    kWrapper << csrRowPtrA << csrColIndA << csrRowPtrB << csrColIndB << fail_perm << csrRowCNnzSize 
        << clhashtable << fail_count << max_intprod << cl::Local(GROUPSIZE_256 * sizeof(cl_int)) 
        << clMaxNnz;

    cl::NDRange local(szLocalWorkSize[0]);
    cl::NDRange global(szGlobalWorkSize[0]);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        CLSPARSE_V(status, "compute_nnzC_tb_global");
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

/**
 * @brief Compute the number of non-zeros of C's rows
 * 
 * @param csrRowPtrA A's row pointer array
 * @param csrColIndA A's column indices array
 * @param csrRowPtrB B's row pointer array
 * @param csrColIndB B's column indices array
 * @param csrRowCReorder Reordered C's rows
 * @param csrRowCNnzSize Buffer for storing the number of non-zeros for each C's row
 * @param innBin Bins of the C's rows with the number of intermediate products
 * @param innPtr Baseline pointers for each bin in csrRowCReorder
 * @param max_intprod the maximum number of intermediate products of C's rows
 * @param clMaxNnz the maximum number of non-zeros of C's rows
 * @param control Control for library
 * @return clsparseStatus If CL_SUCCESS, success, if not, fails.
 */
clsparseStatus compute_nnzC(cl_mem csrRowPtrA, cl_mem csrColIndA, cl_mem csrRowPtrB, cl_mem csrColIndB,
                            cl_mem csrRowCReorder, cl_mem csrRowCNnzSize, int *innBin, int *innPtr, 
                            int max_intprod, cl_mem clMaxNnz, clsparseControl control)
{
    clsparseStatus run_status;
    cl_int status;
    cl_event kernel_events[15];

    if (innBin[0] > 0)
    {
        // rows with 0 intermediate products
        // 256 threads on each workgroup
        run_status = compute_nnzC_0(csrRowCReorder, csrRowCNnzSize, innBin[0], control);

        if (run_status != clsparseSuccess)
            return run_status;
        
        status = ::clRetainEvent(control->event());

        if(status != CL_SUCCESS)
            return (clsparseStatus)status;
            
        kernel_events[0] = control->event();
    }
    if (innBin[1] > 0)
    {
        // rows with 1 intermediate product
        // 256 threads on each workgroup
        run_status = compute_nnzC_1(csrRowCReorder, csrRowCNnzSize, innBin[1], innPtr[1], control);

        if (run_status != clsparseSuccess)
            return run_status;
        
        status = ::clRetainEvent(control->event());

        if(status != CL_SUCCESS)
            return (clsparseStatus)status;
            
        kernel_events[1] = control->event();
    }
    if (innBin[2] > 0)
    {
        // rows with 2 intermediate products
        // 256 threads on each workgroup
        run_status = compute_nnzC_pwarp(csrRowPtrA, csrColIndA, csrRowPtrB, csrColIndB, csrRowCReorder, 
                                        csrRowCNnzSize, innBin[2], innPtr[2], 2, 1, control);

        if (run_status != clsparseSuccess)
            return run_status;
        
        status = ::clRetainEvent(control->event());

        if(status != CL_SUCCESS)
            return (clsparseStatus)status;
            
        kernel_events[2] = control->event();
    }
    if (innBin[3] > 0)
    {
        // rows with 3~4 intermediate products
        // 256 threads on each workgroup
        run_status = compute_nnzC_pwarp(csrRowPtrA, csrColIndA, csrRowPtrB, csrColIndB, csrRowCReorder, 
                                        csrRowCNnzSize, innBin[3], innPtr[3], 4, 2, control);

        if (run_status != clsparseSuccess)
            return run_status;
        
        status = ::clRetainEvent(control->event());

        if(status != CL_SUCCESS)
            return (clsparseStatus)status;
            
        kernel_events[3] = control->event();
    }
    if (innBin[4] > 0)
    {
        // rows with 5~8 intermediate products
        // 256 threads on each workgroup
        run_status = compute_nnzC_pwarp(csrRowPtrA, csrColIndA, csrRowPtrB, csrColIndB, csrRowCReorder, 
                                        csrRowCNnzSize, innBin[4], innPtr[4], 8, 2, control);

        if (run_status != clsparseSuccess)
            return run_status;
        
        status = ::clRetainEvent(control->event());

        if(status != CL_SUCCESS)
            return (clsparseStatus)status;
            
        kernel_events[4] = control->event();
    }
    if (innBin[5] > 0)
    {
        // rows with 9~16 intermediate products
        // 256 threads on each workgroup
        run_status = compute_nnzC_pwarp(csrRowPtrA, csrColIndA, csrRowPtrB, csrColIndB, csrRowCReorder, 
                                        csrRowCNnzSize, innBin[5], innPtr[5], 16, 4, control);

        if (run_status != clsparseSuccess)
            return run_status;
        
        status = ::clRetainEvent(control->event());

        if(status != CL_SUCCESS)
            return (clsparseStatus)status;
            
        kernel_events[5] = control->event();
    }
    if (innBin[6] > 0)
    {
        // rows with 17~32 intermediate products
        // 256 threads on each workgroup
        run_status = compute_nnzC_pwarp(csrRowPtrA, csrColIndA, csrRowPtrB, csrColIndB, csrRowCReorder, 
                                        csrRowCNnzSize, innBin[6], innPtr[6], 32, 4, control);

        if (run_status != clsparseSuccess)
            return run_status;
        
        status = ::clRetainEvent(control->event());

        if(status != CL_SUCCESS)
            return (clsparseStatus)status;
            
        kernel_events[6] = control->event();
    }
    if (innBin[7] > 0)
    {
        // rows with 33~64 intermediate products
        // 256 threads on each workgroup
        run_status = compute_nnzC_pwarp(csrRowPtrA, csrColIndA, csrRowPtrB, csrColIndB, csrRowCReorder, 
                                        csrRowCNnzSize, innBin[7], innPtr[7], 64, 8, control);

        if (run_status != clsparseSuccess)
            return run_status;
        
        status = ::clRetainEvent(control->event());

        if(status != CL_SUCCESS)
            return (clsparseStatus)status;
            
        kernel_events[7] = control->event();
    }
    if (innBin[8] > 0)
    {
        // rows with 65~128 intermediate products
        // 256 threads on each workgroup
        run_status = compute_nnzC_tb(64, csrRowPtrA, csrColIndA, csrRowPtrB, csrColIndB, csrRowCReorder, 
                                     csrRowCNnzSize, innBin[8], innPtr[8], 128, 8, control);

        if (run_status != clsparseSuccess)
            return run_status;
        
        status = ::clRetainEvent(control->event());

        if(status != CL_SUCCESS)
            return (clsparseStatus)status;
            
        kernel_events[8] = control->event();
    }
    if (innBin[9] > 0)
    {
        // rows with 129~256 intermediate products
        // 256 threads on each workgroup
        run_status = compute_nnzC_tb(64, csrRowPtrA, csrColIndA, csrRowPtrB, csrColIndB, csrRowCReorder, 
                                     csrRowCNnzSize, innBin[9], innPtr[9], 256, 16, control);

        if (run_status != clsparseSuccess)
            return run_status;
        
        status = ::clRetainEvent(control->event());

        if(status != CL_SUCCESS)
            return (clsparseStatus)status;
            
        kernel_events[9] = control->event();
    }
    if (innBin[10] > 0)
    {
        // rows with 257~512 intermediate products
        // 64 threads on each workgroup
        run_status = compute_nnzC_tb(128, csrRowPtrA, csrColIndA, csrRowPtrB, csrColIndB, csrRowCReorder, 
                                     csrRowCNnzSize, innBin[10], innPtr[10], 512, 16, control);

        if (run_status != clsparseSuccess)
            return run_status;
        
        status = ::clRetainEvent(control->event());

        if(status != CL_SUCCESS)
            return (clsparseStatus)status;
            
        kernel_events[10] = control->event();
    }
    if (innBin[11] > 0)
    {
        // rows with 513~1024 intermediate products
        // 128 threads on each workgroup
        run_status = compute_nnzC_tb(256, csrRowPtrA, csrColIndA, csrRowPtrB, csrColIndB, csrRowCReorder, 
                                     csrRowCNnzSize, innBin[11], innPtr[11], 1024, 32, control);

        if (run_status != clsparseSuccess)
            return run_status;
        
        status = ::clRetainEvent(control->event());

        if(status != CL_SUCCESS)
            return (clsparseStatus)status;
            
        kernel_events[11] = control->event();
    }
    if (innBin[12] > 0)
    {
        // rows with 1025~2048 intermediate products
        // 256 threads on each workgroup
        run_status = compute_nnzC_tb(256, csrRowPtrA, csrColIndA, csrRowPtrB, csrColIndB, csrRowCReorder, 
                                     csrRowCNnzSize, innBin[12], innPtr[12], 2048, 32, control);

        if (run_status != clsparseSuccess)
            return run_status;
        
        status = ::clRetainEvent(control->event());

        if(status != CL_SUCCESS)
            return (clsparseStatus)status;
            
        kernel_events[12] = control->event();
    }
    if (innBin[13] > 0)
    {
        // rows with 2049~4096 intermediate products
        // 64 threads on each workgroup
        run_status = compute_nnzC_tb(256, csrRowPtrA, csrColIndA, csrRowPtrB, csrColIndB, csrRowCReorder, 
                                     csrRowCNnzSize, innBin[13], innPtr[14], 4096, 64, control);

        if (run_status != clsparseSuccess)
            return run_status;
        
        status = ::clRetainEvent(control->event());

        if(status != CL_SUCCESS)
            return (clsparseStatus)status;
            
        kernel_events[13] = control->event();
    }
    if (innBin[14] > 0)
    {
        // rows with 4096~8192 intermediate products
        // 64 threads on each workgroup
        run_status = compute_nnzC_tb(256, csrRowPtrA, csrColIndA, csrRowPtrB, csrColIndB, csrRowCReorder, 
                                     csrRowCNnzSize, innBin[14], innPtr[14], 8192, 64, control);

        if (run_status != clsparseSuccess)
            return run_status;
        
        status = ::clRetainEvent(control->event());

        if(status != CL_SUCCESS)
            return (clsparseStatus)status;
            
        kernel_events[14] = control->event();
    }
    if (innBin[15] > 0)
    {
        // rows with 8193~ intermediate products
        // 256 threads on each workgroup
        cl::Context cxt = control->getContext();
        int pattern = 0;
        // count  for fail to fit in local memory
        cl_mem clfail_count = ::clCreateBuffer(cxt(), CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, 
                                               sizeof(cl_int), NULL, &status);

        if (status != CL_SUCCESS)
            return (clsparseStatus)status;
        
        // buffer that collects rows that fails to fit in local memory
        cl_mem clfail_perm = ::clCreateBuffer(cxt(), CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, 
                                              innBin[15] * sizeof(cl_int), NULL, &status);

        if (status != CL_SUCCESS)
        {
            ::clReleaseMemObject(clfail_count);
            return (clsparseStatus)status;
        }
            
        cl_event fillEvent;
        // initialize to 0
        status = clEnqueueFillBuffer(control->queue(), clfail_count, &pattern, sizeof(cl_int), 0, 
                                     sizeof(cl_int), 0, NULL, &fillEvent);

        if (status != CL_SUCCESS)
        {
            ::clReleaseMemObject(clfail_perm);
            ::clReleaseMemObject(clfail_count);

            return (clsparseStatus)status;
        }

        status = ::clWaitForEvents(1, &fillEvent);

        if (status != CL_SUCCESS)
        {
            ::clReleaseMemObject(clfail_perm);
            ::clReleaseMemObject(clfail_count);

            return (clsparseStatus)status;
        }

        status = ::clReleaseEvent(fillEvent);

        if (status != CL_SUCCESS)
        {
            ::clReleaseMemObject(clfail_perm);
            ::clReleaseMemObject(clfail_count);

            return (clsparseStatus)status;
        }

        // try to count the number of nonzeros of those large rows
        run_status = compute_nnzC_tb_large(csrRowPtrA, csrColIndA, csrRowPtrB, csrColIndB, 
                                           csrRowCReorder, csrRowCNnzSize, clfail_count, clfail_perm, 
                                           innBin[15], innPtr[15], control);

        if (run_status != CL_SUCCESS)
        {
            ::clReleaseMemObject(clfail_perm);
            ::clReleaseMemObject(clfail_count);

            return run_status;
        }

        status = control->event.wait();

        if (status != CL_SUCCESS)
        {
            ::clReleaseMemObject(clfail_perm);
            ::clReleaseMemObject(clfail_count);

            return (clsparseStatus)status;
        }

        // read the number of rows that failed to count in local memory in CPU
        int fail_count;

        status = clEnqueueReadBuffer(control->queue(), clfail_count, CL_TRUE, 0, sizeof(int), 
                                     &fail_count, 0, NULL, NULL);

        if (status != CL_SUCCESS)
        {
            ::clReleaseMemObject(clfail_perm);
            ::clReleaseMemObject(clfail_count);

            return (clsparseStatus)status;
        }
        // If there is failed rows:
        if (fail_count > 0)
        {
            // allocate hash table on global memory
            int minuspattern = -1;
            cl_mem clhashtable = ::clCreateBuffer(cxt(), CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, 
                                                  max_intprod * fail_count * sizeof(cl_int), 
                                                  NULL, &status);

            if (status != CL_SUCCESS)
            {
                ::clReleaseMemObject(clfail_perm);
                ::clReleaseMemObject(clfail_count);

                return (clsparseStatus)status;
            }

            cl_event hash_fill_event;

            // initialize hash table as -1
            status = clEnqueueFillBuffer(control->queue(), clfail_perm, &minuspattern, sizeof(cl_int), 0, 
                                         max_intprod * fail_count * sizeof(cl_int), 0, NULL, &hash_fill_event);

            if (status != CL_SUCCESS)
            {
                ::clReleaseMemObject(clhashtable);
                ::clReleaseMemObject(clfail_perm);
                ::clReleaseMemObject(clfail_count);

                return (clsparseStatus)status;
            }

            status = ::clWaitForEvents(1, &hash_fill_event);

            if (status != CL_SUCCESS)
            {
                ::clReleaseMemObject(clhashtable);
                ::clReleaseMemObject(clfail_perm);
                ::clReleaseMemObject(clfail_count);

                return (clsparseStatus)status;
            }

            status = ::clReleaseEvent(hash_fill_event);

            if (status != CL_SUCCESS)
            {
                ::clReleaseMemObject(clhashtable);
                ::clReleaseMemObject(clfail_perm);
                ::clReleaseMemObject(clfail_count);

                return (clsparseStatus)status;
            }

            // count the number of non-zeros on global memory
            run_status = compute_nnzC_tb_global(csrRowPtrA, csrColIndA, csrRowPtrB, csrColIndB,
                                                clfail_perm, csrRowCNnzSize, clhashtable, fail_count, 
                                                max_intprod, clMaxNnz, control);

            if (run_status != CL_SUCCESS)
            {
                ::clReleaseMemObject(clhashtable);
                ::clReleaseMemObject(clfail_perm);
                ::clReleaseMemObject(clfail_count);

                return run_status;
            }

            status = control->event.wait();

            if (status != CL_SUCCESS)
            {
                ::clReleaseMemObject(clhashtable);
                ::clReleaseMemObject(clfail_perm);
                ::clReleaseMemObject(clfail_count);

                return run_status;
            }

            // free hash table
            status = ::clReleaseMemObject(clhashtable);

            if (status != CL_SUCCESS)
            {
                ::clReleaseMemObject(clfail_perm);
                ::clReleaseMemObject(clfail_count);

                return (clsparseStatus)status;
            }
        }
        // free failed row buffer
        status = ::clReleaseMemObject(clfail_perm);

        if (status != CL_SUCCESS)
        {
            ::clReleaseMemObject(clfail_count);

            return (clsparseStatus)status;
        }
        // free failed row count
        status = ::clReleaseMemObject(clfail_count);

        if (status != CL_SUCCESS)
            return (clsparseStatus)status;
    }

    for(int i = 0; i < 15; i++)
    {
        if(innBin[i] > 0)
        {
            status = ::clWaitForEvents(1, &kernel_events[i]);

            if(status != CL_SUCCESS)
                return (clsparseStatus)status;
        }
    }

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

/**
 * @brief Perform Prefix-scan for constructing C's row pointer array
 * 
 * @param _m The number of C's rows
 * @param d_array Array that will be prefix-scanned
 * @param control Control for library
 * @return clsparseStatus If CL_SUCCESS, success, if not, fails.
 */
clsparseStatus compute_scan(int _m, cl_mem d_array, clsparseControl control)
{
    const std::string params = std::string() + "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type 
        + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type;

    cl::Kernel scankernel = KernelCache::get(control->queue, "SpGEMM_scan_kernels", "scan_block", params);

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    int num_threads = GROUPSIZE_256;

    size_t num_blocks = ceil((double)_m / (double)GROUPSIZE_256);

    szLocalWorkSize[0] = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    // allocate sum buffer that stores the last value of each block
    int pattern = 0;

    cl_int run_status;
    cl::Context cxt = control->getContext();
    // Allocate a summation buffer
    cl_mem d_sum = ::clCreateBuffer(cxt(), CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, 
                                    num_blocks * sizeof(cl_int), NULL, &run_status);

    if (run_status != CL_SUCCESS)
    {
        CLSPARSE_V(run_status, "compute_scan, clCreateBuffer");
        return (clsparseStatus)run_status;
    }
        
    // perform prefix-scan in each block
    KernelWrap scankWrapper(scankernel);
    scankWrapper << d_array << cl::Local(2 * GROUPSIZE_256 * sizeof(int)) << d_sum << _m;

    cl::NDRange local(szLocalWorkSize[0]);
    cl::NDRange global(szGlobalWorkSize[0]);

    cl_int status = scankWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        CLSPARSE_V(status, "compute_scan, scan");
        ::clReleaseMemObject(d_sum);

        return (clsparseStatus)status;
    }

    status = control->event.wait();

    if (status != CL_SUCCESS)
    {
        CLSPARSE_V(status, "compute_scan, control->event.wait()");
        ::clReleaseMemObject(d_sum);

        return (clsparseStatus)status;
    }

    // if there are multiple sums
    if (_m > 1)
    {
        // prefix-scan the sum block
        clsparseStatus sparsestatus = compute_scan(num_blocks, d_sum, control);

        if (sparsestatus != clsparseSuccess)
        {
            CLSPARSE_V(sparsestatus, "compute_scan, compute_scan");
            ::clReleaseMemObject(d_sum);

            return (clsparseStatus)status;
        }
    }

    cl::Kernel addkernel = KernelCache::get(control->queue, "SpGEMM_scan_kernels", "add_block", params);
    // add the scanned sums into each block
    KernelWrap addkWrapper(addkernel);
    addkWrapper << d_array << d_sum << _m;

    cl_int status = addkWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        CLSPARSE_V(status, "compute_scan, add");
        ::clReleaseMemObject(d_sum);

        return (clsparseStatus)status;
    }

    status = control->event.wait();

    if (status != CL_SUCCESS)
    {
        CLSPARSE_V(status, "compute_scan, control->event.wait()");
        ::clReleaseMemObject(d_sum);

        return (clsparseStatus)status;
    }

    // release the sum buffer
    run_status = ::clReleaseMemObject(d_sum);

    if (run_status != CL_SUCCESS)
    {
        CLSPARSE_V(status, "compute_scan, clReleaseMemObject");
        ::clReleaseMemObject(d_sum);

        return (clsparseStatus)status;
    }

    return clsparseSuccess;
}

/**
 * @brief Count the number of rows in each the number of non-zeros' bin
 * 
 * @param _m The number of C's rows
 * @param csrRowCNnzSize Array of the number of non-zeros of each C's row
 * @param nnzBin Bins of C's rows
 * @param control Control for library
 * @return clsparseStatus If CL_SUCCESS, success, if not, fails.
 */
clsparseStatus compute_nnzBin(int _m, cl_mem csrRowCNnzSize, cl_mem nnzBin, clsparseControl control)
{
    const std::string params = std::string() + "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type 
        + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type;

    cl::Kernel kernel = KernelCache::get(control->queue, "SpGEMM_computeNipBin_kernels", 
                                         "compute_nipBin_kernel", params);

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    int num_threads = GROUPSIZE_256;
    size_t num_blocks = ceil((double)_m / (double)num_threads);

    szLocalWorkSize[0] = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    KernelWrap kWrapper(kernel);

    kWrapper << csrRowCNnzSize << cl::Local(NNZ_SEGMENTS * sizeof(int)) << nnzBin << _m;

    cl::NDRange local(szLocalWorkSize[0]);
    cl::NDRange global(szGlobalWorkSize[0]);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        CLSPARSE_V(status, "compute_nnzBin");
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

/**
 * @brief reorder the rows in bins with the number of non-zeros
 * 
 * @param _m The number of C's rows
 * @param csrRowCNnzSize Array of the number of non-zeros of each C's row
 * @param clNnzPtr Baseline pointers of each bin
 * @param csrRowCReorder Reordered C's rows
 * @param control Control for library
 * @return clsparseStatus If CL_SUCCESS, success, if not, fails.
 */
clsparseStatus compute_reorderRowNnz(int _m, cl_mem csrRowCNnzSize, cl_mem clNnzPtr, 
                                     cl_mem csrRowCReorder, clsparseControl control)
{
    const std::string params = std::string() + "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type 
        + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type;

    cl::Kernel kernel = KernelCache::get(control->queue, "SpGEMM_computeReorderRowNip_kernels", 
                                         "compute_ReorderRowNip_kernel", params);

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    int num_threads = GROUPSIZE_256;
    size_t num_blocks = ceil((double)_m / (double)num_threads);

    szLocalWorkSize[0] = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    KernelWrap kWrapper(kernel);

    kWrapper << csrRowCNnzSize << clNnzPtr << csrRowCReorder << _m;

    cl::NDRange local(szLocalWorkSize[0]);
    cl::NDRange global(szGlobalWorkSize[0]);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        CLSPARSE_V(status, "compute_reorderRowNnz");
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

/**
 * @brief Compute columns and values of C's rows with 1 non-zero
 * 
 * @param csrRowPtrA A's row pointer array
 * @param csrColIndA A's column indices array
 * @param csrValA  A's value array
 * @param csrRowPtrB B's row pointer array
 * @param csrColIndB B's column indices array
 * @param csrValB B's value array
 * @param csrRowPtrC C's row pointer array
 * @param csrColIndC C's column indices array
 * @param csrValC C's value array
 * @param csrRowCReorder Reordered C's rows
 * @param bin Number of C's rows with 1 non-zero
 * @param ptr Baseline pointer for the non-zero bin
 * @param control Control for library
 * @return clsparseStatus If CL_SUCCESS, success, if not, fails.
 */
clsparseStatus compute_valC_1(cl_mem csrRowPtrA, cl_mem csrColIndA, cl_mem csrValA, cl_mem csrRowPtrB,
                              cl_mem csrColIndB, cl_mem csrValB, cl_mem csrRowPtrC, cl_mem csrColIndC, 
                              cl_mem csrValC, cl_mem csrRowCReorder, int bin, int ptr, 
                              clsparseControl control)
{
    const std::string params = std::string() + "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type 
        + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type;

    cl::Kernel kernel = KernelCache::get(control->queue, "SpGEMM_computeVal_kernels", "compute_valC_1", 
        params);

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    int num_threads = GROUPSIZE_256;
    size_t num_blocks = ceil((double)bin / (double)num_threads);

    szLocalWorkSize[0] = GROUPSIZE_256;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    KernelWrap kWrapper(kernel);
    kWrapper << csrRowPtrA << csrColIndA << csrValA << csrRowPtrB << csrColIndB << csrValB << csrRowPtrC 
        << csrColIndC << csrValC << csrRowCReorder << bin << ptr;

    cl::NDRange local(szLocalWorkSize[0]);
    cl::NDRange global(szGlobalWorkSize[0]);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        CLSPARSE_V(status, "compute_val_1");
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

/**
 * @brief Compute columns and values of C's rows with 2~64 non-zeros
 * 
 * @param csrRowPtrA A's row pointer array
 * @param csrColIndA A's column indices array
 * @param csrValA  A's value array
 * @param csrRowPtrB B's row pointer array
 * @param csrColIndB B's column indices array
 * @param csrValB B's value array
 * @param csrRowPtrC C's row pointer array
 * @param csrColIndC C's column indices array
 * @param csrValC C's value array
 * @param csrRowCReorder Reordered C's rows
 * @param bin Number of C's rows with 1 non-zero
 * @param ptr Baseline pointer for the non-zero bin
 * @param nnzSize maximum non-zeros in each bin
 * @param control Control for library
 * @return clsparseStatus If CL_SUCCESS, success, if not, fails.
 */
clsparseStatus compute_valC_pwarp(cl_mem csrRowPtrA, cl_mem csrColIndA, cl_mem csrValA, 
                                  cl_mem csrRowPtrB, cl_mem csrColIndB, cl_mem csrValB, 
                                  cl_mem csrRowPtrC, cl_mem csrColIndC, cl_mem csrValC, 
                                  cl_mem csrRowCReorder, int bin, int ptr, int nnzSize, 
                                  clsparseControl control)
{
    const std::string params = std::string() + "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type 
        + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type;

    cl::Kernel kernel = KernelCache::get(control->queue, "SpGEMM_computeVal_kernels", 
                                         "compute_valC_pwarp", params);

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    int num_threads = 64;
    size_t num_blocks = ceil((double)bin / (double)num_threads);

    szLocalWorkSize[0] = 64;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    KernelWrap kWrapper(kernel);
    kWrapper << csrRowPtrA << csrColIndA << csrValA << csrRowPtrB << csrColIndB << csrValB << csrRowPtrC 
        << csrColIndC << csrValC << csrRowCReorder  << cl::Local((64 / nnzSize) * sizeof(cl_int)) << bin 
        << ptr << nnzSize;

    cl::NDRange local(szLocalWorkSize[0]);
    cl::NDRange global(szGlobalWorkSize[0]);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        CLSPARSE_V(status, "compute_valC_pwarp");
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

/**
 * @brief Compute columns and values of C's rows with 65~2048 non-zeros
 * 
 * @param num_threads Number of threads for each workgroup
 * @param csrRowPtrA A's row pointer array
 * @param csrColIndA A's column indices array
 * @param csrValA  A's value array
 * @param csrRowPtrB B's row pointer array
 * @param csrColIndB B's column indices array
 * @param csrValB B's value array
 * @param csrRowPtrC C's row pointer array
 * @param csrColIndC C's column indices array
 * @param csrValC C's value array
 * @param csrRowCReorder Reordered C's rows
 * @param bin Number of C's rows with 1 non-zero
 * @param ptr Baseline pointer for the non-zero bin
 * @param nnzSize maximum non-zeros in each bin
 * @param control Control for library
 * @return clsparseStatus If CL_SUCCESS, success, if not, fails.
 */
clsparseStatus compute_valC_tb(int num_threads, cl_mem csrRowPtrA, cl_mem csrColIndA, cl_mem csrValA,
                               cl_mem csrRowPtrB, cl_mem csrColIndB, cl_mem csrValB, cl_mem csrRowPtrC, 
                               cl_mem csrColIndC, cl_mem csrValC,  cl_mem csrRowCReorder, int bin, 
                               int ptr, int nnzSize, clsparseControl control)
{
    const std::string params = std::string() + "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type 
        + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type;

    cl::Kernel kernel = KernelCache::get(control->queue, "SpGEMM_computeVal_kernels", "compute_valC_tb", 
                                         params);

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    size_t num_blocks = ceil((double)bin / (double)num_threads);

    szLocalWorkSize[0] = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    KernelWrap kWrapper(kernel);
    kWrapper << csrRowPtrA << csrColIndA << csrValA << csrRowPtrB << csrColIndB << csrValB << csrRowPtrC 
        << csrColIndC << csrValC << csrRowCReorder << cl::Local(sizeof(cl_int)) 
        << cl::Local((2 * nnzSize - 1) * sizeof(int)) << cl::Local((2 * nnzSize - 1) * sizeof(int)) 
        << bin << ptr << nnzSize;

    cl::NDRange local(szLocalWorkSize[0]);
    cl::NDRange global(szGlobalWorkSize[0]);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        CLSPARSE_V(status, "compute_valC_tb");
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

/**
 * @brief Compute columns and values of C's rows with 2049~ non-zeros
 * 
 * @param csrRowPtrA A's row pointer array
 * @param csrColIndA A's column indices array
 * @param csrValA  A's value array
 * @param csrRowPtrB B's row pointer array
 * @param csrColIndB B's column indices array
 * @param csrValB B's value array
 * @param csrRowPtrC C's row pointer array
 * @param csrColIndC C's column indices array
 * @param csrValC C's value array
 * @param csrRowCReorder Reordered C's rows
 * @param clcolhashtable Global memory buffer for computing column indices
 * @param clvalhashtable Global memory buffer for computing values
 * @param bin Number of C's rows with 1 non-zero
 * @param ptr Baseline pointer for the non-zero bin
 * @param nnzSize maximum non-zeros in each bin
 * @param control Control for library
 * @return clsparseStatus If CL_SUCCESS, success, if not, fails.
 */
clsparseStatus compute_valC_tb_global(cl_mem csrRowPtrA, cl_mem csrColIndA, cl_mem csrValA, 
                                      cl_mem csrRowPtrB, cl_mem csrColIndB, cl_mem csrValB, 
                                      cl_mem csrRowPtrC, cl_mem csrColIndC, cl_mem csrValC, 
                                      cl_mem csrRowCReorder, cl_mem clcolhashtable, 
                                      cl_mem clvalhashtable, int bin, int ptr, int nnzSize,
                                      clsparseControl control)
{
    const std::string params = std::string() + "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type 
        + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type;

    cl::Kernel kernel = KernelCache::get(control->queue, "SpGEMM_computeVal_kernels", 
                                         "compute_valC_tb_global", params);

    int num_threads = GROUPSIZE_256;

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    size_t num_blocks = ceil((double)bin / (double)num_threads);

    szLocalWorkSize[0] = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    KernelWrap kWrapper(kernel);
    kWrapper << csrRowPtrA << csrColIndA << csrValA << csrRowPtrB << csrColIndB << csrValB << csrRowPtrC 
        << csrColIndC << csrValC << csrRowCReorder << cl::Local(sizeof(cl_int)) << clcolhashtable 
        << clvalhashtable << bin << ptr << nnzSize;

    cl::NDRange local(szLocalWorkSize[0]);
    cl::NDRange global(szGlobalWorkSize[0]);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        CLSPARSE_V(status, "compute_valC_tb_global");
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

// compute the value of rows with non-zero elements
/**
 * @brief Compute columns and values of C's rows
 * 
 * @param csrRowPtrA A's row pointer array
 * @param csrColIndA A's column indices array
 * @param csrValA  A's value array
 * @param csrRowPtrB B's row pointer array
 * @param csrColIndB B's column indices array
 * @param csrValB B's value array
 * @param csrRowPtrC C's row pointer array
 * @param csrColIndC C's column indices array
 * @param csrValC C's value array
 * @param csrRowCReorder Reordered C's rows
 * @param nnzBin Bins of C's rows with the number of non-zeros
 * @param nnzPtr 
 * @param max_nnz 
 * @param control 
 * @return clsparseStatus 
 */
clsparseStatus compute_valC(cl_mem csrRowPtrA, cl_mem csrColIndA, cl_mem csrValA, cl_mem csrRowPtrB,
                            cl_mem csrColIndB, cl_mem csrValB, cl_mem csrRowPtrC, cl_mem csrColIndC, 
                            cl_mem csrValC, cl_mem csrRowCReorder, int *nnzBin, int *nnzPtr, int max_nnz, 
                            clsparseControl control)
{  
    clsparseStatus run_status;
    cl_int status;
    cl_event kernel_events[12];

    if (nnzBin[1] > 0)
    {
        // compute values with 1 non-zero element
        run_status = compute_valC_1(csrRowPtrA, csrColIndA, csrValA, csrRowPtrB, csrColIndB, 
                                    csrValB, csrRowPtrC, csrColIndC, csrValC, csrRowCReorder, 
                                    nnzBin[1], nnzPtr[1], control);

        if (run_status != clsparseSuccess)
            return run_status;
        
        status = ::clRetainEvent(control->event());

        if(status != CL_SUCCESS)
            return (clsparseStatus)status;

        kernel_events[0] = control->event();
    }

    int i;

    for (i = 2; i <= 6; i++)
    {
        if (nnzBin[i] > 0)
        {
            // compute values with {2, 3~4, 5~8, 9~16, 17~32} non-zero elements
            run_status = compute_valC_pwarp(csrRowPtrA, csrColIndA, csrValA, csrRowPtrB, csrColIndB, 
                                            csrValB, csrRowPtrC, csrColIndC, csrValC, 
                                            csrRowCReorder, nnzBin[i], nnzPtr[i], 1 << (i - 1), control);
            if (run_status != clsparseSuccess)
                return run_status;
            
            status = ::clRetainEvent(control->event());

            if(status != CL_SUCCESS)
                return (clsparseStatus)status;
                
            kernel_events[i - 1] = control->event();
        }
    }
    for (i = 7; i < 10; i++)
    {
        if (nnzBin[i] > 0)
        {
            // compute values with {33~64, 65~128, 129~256} non-zero elements
            run_status = compute_valC_tb(1 << (i - 1), csrRowPtrA, csrColIndA, csrValA, csrRowPtrB, 
                                         csrColIndB, csrValB, csrRowPtrC, csrColIndC, csrValC, 
                                         csrRowCReorder, nnzBin[i], nnzPtr[i], 1 << (i - 1), control);
            if (run_status != clsparseSuccess)
                return run_status;
            
            status = ::clRetainEvent(control->event());

            if(status != CL_SUCCESS)
                return (clsparseStatus)status;
                
            kernel_events[i - 1] = control->event();
        }
    }

    for (i = 10; i < 12; i++)
    {
        if (nnzBin[i] > 0)
        {
            // compute values with {257~512, 513~1024, 1025~2048} non-zero elements
            int num_threads = GROUPSIZE_256;

            run_status = compute_valC_tb(num_threads, csrRowPtrA, csrColIndA, csrValA, csrRowPtrB, 
                                         csrColIndB, csrValB, csrRowPtrC, csrColIndC, csrValC, 
                                         csrRowCReorder, nnzBin[i], nnzPtr[i], 1 << (i - 1), control);

            if (run_status != clsparseSuccess)
                return run_status;
            
            status = ::clRetainEvent(control->event());

            if(status != CL_SUCCESS)
                return (clsparseStatus)status;
                
            kernel_events[i - 1] = control->event();
        }
    }
    if (nnzBin[13] > 0)
    {
        // compute values with 2049~4096 non-zero elements
        int pattern = 0;
        int minuspattern = -1;
        cl::Context cxt = control->getContext();
        // allocate column hash table with size 8192 * sizeof(cl_int)
        cl_mem clcolhashtable = ::clCreateBuffer(cxt(), CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, 
                                                 8192 * sizeof(cl_int), NULL, &status);

        if (status != CL_SUCCESS)
            return (clsparseStatus)status;
        
        // allocate value hash table with size 8192 * sizeof(cl_int)
        cl_mem clvalhashtable = ::clCreateBuffer(cxt(), CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, 
                                                 8192 * sizeof(cl_int), NULL, &status);

        if (status != CL_SUCCESS)
        {
            ::clReleaseMemObject(clcolhashtable);

            return (clsparseStatus)status;
        }

        cl_event fill_events[2];
    
        // initialize column hash table as -1
        status = clEnqueueFillBuffer(control->queue(), clcolhashtable, &minuspattern, sizeof(cl_int), 0, 
                                     nnzBin[13] * sizeof(cl_int), 0, NULL, &fill_events[0]);

        if (status != CL_SUCCESS)
        {
            ::clReleaseMemObject(clvalhashtable);
            ::clReleaseMemObject(clcolhashtable);

            return (clsparseStatus)status;
        }
        
        // initialize value hash table as 0
        status = clEnqueueFillBuffer(control->queue(), clvalhashtable, &pattern, sizeof(cl_int), 0, 
                                     nnzBin[13] * sizeof(cl_int), 0, NULL, &fill_events[1]);

        if (status != CL_SUCCESS)
        {
            ::clReleaseMemObject(clvalhashtable);
            ::clReleaseMemObject(clcolhashtable);

            return (clsparseStatus)status;
        }

        status = ::clWaitForEvents(2, fill_events);

        if (status != CL_SUCCESS)
        {
            ::clReleaseMemObject(clvalhashtable);
            ::clReleaseMemObject(clcolhashtable);

            return (clsparseStatus)status;
        }

        status = ::clReleaseEvent(fill_events[0]);

        if (status != CL_SUCCESS)
        {
            ::clReleaseMemObject(clvalhashtable);
            ::clReleaseMemObject(clcolhashtable);

            return (clsparseStatus)status;
        }

        status = ::clReleaseEvent(fill_events[1]);

        if (status != CL_SUCCESS)
        {
            ::clReleaseMemObject(clvalhashtable);
            ::clReleaseMemObject(clcolhashtable);

            return (clsparseStatus)status;
        }
        // compute values on global hash table
        run_status = compute_valC_tb_global(csrRowPtrA, csrColIndA, csrValA, csrRowPtrB, csrColIndB, 
                                            csrValB, csrRowPtrC, csrColIndC, csrValC, csrRowCReorder, 
                                            clcolhashtable, clvalhashtable, nnzBin[i], nnzPtr[i], 4096, 
                                            control);

        if (run_status != CL_SUCCESS)
        {
            ::clReleaseMemObject(clvalhashtable);
            ::clReleaseMemObject(clcolhashtable);

            return run_status;
        }

        status = control->event.wait();

        if (status != CL_SUCCESS)
        {
            ::clReleaseMemObject(clvalhashtable);
            ::clReleaseMemObject(clcolhashtable);

            return (clsparseStatus)status;
        }
        // free value hash table
        status = ::clReleaseMemObject(clvalhashtable);

        if (status != CL_SUCCESS)
        {
            ::clReleaseMemObject(clcolhashtable);

            return (clsparseStatus)status;
        }
        // free column hash table
        status = ::clReleaseMemObject(clcolhashtable);

        if (status != CL_SUCCESS)
            return (clsparseStatus)status;
    }
    if (nnzBin[14] > 0)
    {
        // compute values with 4097~ non-zero elements
        int pattern = 0, minuspattern = -1;
        cl::Context cxt = control->getContext();
        // allocate column hash table with size 2 * max_nnz * sizeof(cl_int)
        cl_mem clcolhashtable = ::clCreateBuffer(cxt(), CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, 
                                                 2 * max_nnz * sizeof(cl_int), NULL, &status);

        if (status != CL_SUCCESS)
            return (clsparseStatus)status;
        // allocate value hash table with size 2 * max_nnz * sizeof(cl_int)
        cl_mem clvalhashtable = ::clCreateBuffer(cxt(), CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, 
                                                 2 * max_nnz * sizeof(cl_int), NULL, &status);

        if (status != CL_SUCCESS)
        {
            ::clReleaseMemObject(clcolhashtable);

            return (clsparseStatus)status;
        }
        
        cl_event fill_events[2];
        // initialize column hash table as -1
        status = clEnqueueFillBuffer(control->queue(), clcolhashtable, &minuspattern, sizeof(cl_int), 0, 
                                     2 * max_nnz * sizeof(cl_int), 0, NULL, &fill_events[0]);

        if (status != CL_SUCCESS)
        {
            ::clReleaseMemObject(clvalhashtable);
            ::clReleaseMemObject(clcolhashtable);

            return (clsparseStatus)status;
        }
        // initialize value hash table as 0
        status = clEnqueueFillBuffer(control->queue(), clvalhashtable, &pattern, sizeof(cl_int), 0, 
                                         2 * max_nnz * sizeof(cl_int), 0, NULL, &fill_events[1]);

        if (status != CL_SUCCESS)
        {
            ::clReleaseMemObject(clvalhashtable);
            ::clReleaseMemObject(clcolhashtable);

            return (clsparseStatus)status;
        }

        status = ::clWaitForEvents(2, fill_events);

        if (status != CL_SUCCESS)
        {
            ::clReleaseMemObject(clvalhashtable);
            ::clReleaseMemObject(clcolhashtable);

            return (clsparseStatus)status;
        }

        status = ::clReleaseEvent(fill_events[0]);

        if (status != CL_SUCCESS)
        {
            ::clReleaseMemObject(clvalhashtable);
            ::clReleaseMemObject(clcolhashtable);

            return (clsparseStatus)status;
        }

        status = ::clReleaseEvent(fill_events[1]);

        if (status != CL_SUCCESS)
        {
            ::clReleaseMemObject(clvalhashtable);
            ::clReleaseMemObject(clcolhashtable);

            return (clsparseStatus)status;
        }

        // compute values on global hash table
        run_status = compute_valC_tb_global(csrRowPtrA, csrColIndA, csrValA, csrRowPtrB, csrColIndB, 
                                            csrValB, csrRowPtrC, csrColIndC, csrValC, csrRowCReorder, 
                                            clcolhashtable, clvalhashtable, nnzBin[i], nnzPtr[i], 
                                            max_nnz, control);

        if (run_status != CL_SUCCESS)
        {
            ::clReleaseMemObject(clvalhashtable);
            ::clReleaseMemObject(clcolhashtable);

            return run_status;
        }

        status = control->event.wait();

        if (run_status != CL_SUCCESS)
        {
            ::clReleaseMemObject(clvalhashtable);
            ::clReleaseMemObject(clcolhashtable);

            return run_status;
        }

        // free value hash table
        status = ::clReleaseMemObject(clvalhashtable);

        if (status != CL_SUCCESS)
        {
            ::clReleaseMemObject(clcolhashtable);

            return (clsparseStatus)status;
        }
        // free column hash table
        status = ::clReleaseMemObject(clcolhashtable);

        if (status != CL_SUCCESS)
            return (clsparseStatus)status;
    }

    for(int i = 0; i < 12; i++)
    {
        if(nnzBin[i + 1] > 0)
        {
            status = ::clWaitForEvents(1, &kernel_events[i]);

            if (status != CL_SUCCESS)
                return (clsparseStatus)status;
        }
    }

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

// C = A * B, A, B, C are CSR Matrix
/*CLSPARSE_EXPORT*/ clsparseStatus
clsparseScsrSpGemm(
    const clsparseCsrMatrix *sparseMatA,
    const clsparseCsrMatrix *sparseMatB,
    clsparseCsrMatrix *sparseMatC,
    const clsparseControl control)
{
    cl_int run_status;

    if (!clsparseInitialized)
    {
        return clsparseNotInitialized;
    }

    if (control == nullptr)
    {
        return clsparseInvalidControlObject;
    }

    const clsparseCsrMatrixPrivate *matA = static_cast<const clsparseCsrMatrixPrivate *>(sparseMatA);
    const clsparseCsrMatrixPrivate *matB = static_cast<const clsparseCsrMatrixPrivate *>(sparseMatB);
    clsparseCsrMatrixPrivate *matC = static_cast<clsparseCsrMatrixPrivate *>(sparseMatC);

    size_t m = matA->num_rows;
    size_t k1 = matA->num_cols;
    size_t k2 = matB->num_rows;
    size_t n = matB->num_cols;
    size_t nnzA = matA->num_nonzeros;
    size_t nnzB = matB->num_nonzeros;

    if (k1 != k2)
    {
        std::cerr << "A.n and B.m don't match!" << std::endl;
        return clsparseInvalidKernelExecution;
    }

    cl_mem csrRowPtrA = matA->row_pointer;
    cl_mem csrColIndA = matA->col_indices;
    cl_mem csrValA = matA->values;
    cl_mem csrRowPtrB = matB->row_pointer;
    cl_mem csrColIndB = matB->col_indices;
    cl_mem csrValB = matB->values;

    cl::Context cxt = control->getContext();

    // STAGE 1 Count the number of intermediate products of each row
    int pattern = 0;

    // intermediate product numbers of each C's row, m integers.
    cl_mem csrRowCInnProdNum = ::clCreateBuffer(cxt(), CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, 
                                                m * sizeof(cl_int), NULL, &run_status);

    if (run_status != CL_SUCCESS)
    {
        CLSPARSE_V(run_status, "csrRowCInnProdNum = clCreateBuffer()");
        return (clsparseStatus)run_status;
    }
    
    // maximum inner-product number of all rows
    cl_mem clMaxInnProd = ::clCreateBuffer(cxt(), CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, 
                                           sizeof(cl_int), NULL, &run_status);

    if (run_status != CL_SUCCESS)
    {
        CLSPARSE_V(run_status, "clMaxInnProd = clCreateBuffer");
        ::clReleaseMemObject(csrRowCInnProdNum);

        return (clsparseStatus)run_status;
    }

    cl_event innprod_event[2];

    // Fill csrRowCInnProdNum as 0
    run_status = clEnqueueFillBuffer(control->queue(), csrRowCInnProdNum, &pattern, sizeof(cl_int), 0, 
                                     m * sizeof(cl_int), 0, NULL, &innprod_event[0]);

    if (run_status != CL_SUCCESS)
    {
        CLSPARSE_V(run_status, "clEnqueueFillBuffer(&innprod_event[0])")
        ::clReleaseMemObject(clMaxInnProd);
        ::clReleaseMemObject(csrRowCInnProdNum);

        return (clsparseStatus)run_status;
    }

    // initialize to 0
    run_status = clEnqueueFillBuffer(control->queue(), clMaxInnProd, &pattern, sizeof(cl_int), 0, 
                                     sizeof(cl_int), 0, NULL, &innprod_event[1]);

    if (run_status != CL_SUCCESS)
    {
        CLSPARSE_V(run_status, "clEnqueueFillBuffer(&innprod_event[1]");
        ::clReleaseMemObject(clMaxInnProd);
        ::clReleaseMemObject(csrRowCInnProdNum);

        return (clsparseStatus)run_status;
    }

    run_status = ::clWaitForEvents(2, innprod_event);

    if (run_status != CL_SUCCESS)
    {
        CLSPARSE_V(run_status, "clWaitForEvents(innprod_event)");
        ::clReleaseMemObject(clMaxInnProd);
        ::clReleaseMemObject(csrRowCInnProdNum);

        return (clsparseStatus)run_status;
    }

    run_status = ::clReleaseEvent(innprod_event[0]);

    if (run_status != CL_SUCCESS)
    {
        CLSPARSE_V(run_status, "clReleaseEvent(innprod_event[0])");
        ::clReleaseMemObject(clMaxInnProd);
        ::clReleaseMemObject(csrRowCInnProdNum);

        return (clsparseStatus)run_status;
    }

    run_status = ::clReleaseEvent(innprod_event[1]);

    if (run_status != CL_SUCCESS)
    {
        CLSPARSE_V(run_status, "clReleaseEvent(innprod_event[1]");
        ::clReleaseMemObject(clMaxInnProd);
        ::clReleaseMemObject(csrRowCInnProdNum);

        return (clsparseStatus)run_status;
    }

    // calculate the numbers of inner-product
    run_status = compute_CInnProdNum(m, csrRowPtrA, csrColIndA, csrRowPtrB, csrRowCInnProdNum, 
                                        clMaxInnProd, control);

    if (run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(clMaxInnProd);
        ::clReleaseMemObject(csrRowCInnProdNum);

        return (clsparseStatus)run_status;
    }

    run_status = control->event.wait();

    if (run_status != CL_SUCCESS)
    {
        CLSPARSE_V(run_status, "control->event.wait()");
        ::clReleaseMemObject(clMaxInnProd);
        ::clReleaseMemObject(csrRowCInnProdNum);

        return (clsparseStatus)run_status;
    }

    int max_intprod;
    // retrieve the maximum inner-product number to CPU.
    run_status = clEnqueueReadBuffer(control->queue(), clMaxInnProd, CL_TRUE, 0, sizeof(int), 
                                     &max_intprod, 0, NULL, NULL);

    if (run_status != CL_SUCCESS)
    {
        CLSPARSE_V(run_status, "clEnqueueReadBuffer(clMaxINnProd, &max_intprod)");
        ::clReleaseMemObject(clMaxInnProd);
        ::clReleaseMemObject(csrRowCInnProdNum);

        return (clsparseStatus)run_status;
    }
    // clMaxInnProd will not be used again: free!
    run_status = ::clReleaseMemObject(clMaxInnProd);

    if (run_status != CL_SUCCESS)
    {
        CLSPARSE_V(run_status, "clReleaseMemObject(clMaxInnProd)")
        ::clReleaseMemObject(csrRowCInnProdNum);

        return (clsparseStatus)run_status;
    }

    // STAGE 2 Divide the rows into groups by the number of intermediate products
    // bins of rows with inner-product numbers
    // local memory size: 32KB.
    // 16 members of bin: {0, 1, 2, 3~4, 5~8, 9~16, 17~32, 33~64, 65~128, 129~256, 257~512, 513~1024, 
    // 1025~2048, 2049~4096, 4097~8192, 8192~}
    cl_mem clinnBin = ::clCreateBuffer(cxt(), CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, 
                                       NIP_SEGMENTS * sizeof(cl_int), NULL, &run_status);

    if (run_status != CL_SUCCESS)
    {
        CLSPARSE_V(run_status, "clinnBin = clCreateBuffer()")
        ::clReleaseMemObject(csrRowCInnProdNum);

        return (clsparseStatus)run_status;
    }

    cl_event nipbin_event;

    run_status = clEnqueueFillBuffer(control->queue(), clinnBin, &pattern, sizeof(cl_int), 0, 
                                     NIP_SEGMENTS * sizeof(cl_int), 0, NULL, &nipbin_event);

    if (run_status != CL_SUCCESS)
    {
        CLSPARSE_V(run_status, "clEnqueueFillBuffer(&nipbin_event)");
        ::clReleaseMemObject(clinnBin);
        ::clReleaseMemObject(csrRowCInnProdNum);

        return (clsparseStatus)run_status;
    }

    run_status = ::clWaitForEvents(1, &nipbin_event);

    if (run_status != CL_SUCCESS)
    {
        CLSPARSE_V(run_status, "clWaitForEvents(&nipbin_event)");
        ::clReleaseMemObject(clinnBin);
        ::clReleaseMemObject(csrRowCInnProdNum);

        return (clsparseStatus)run_status;
    }

    run_status = ::clReleaseEvent(nipbin_event);

    if (run_status != CL_SUCCESS)
    {
        CLSPARSE_V(run_status, "clReleaseEvent(nipbin_event)");
        ::clReleaseMemObject(clinnBin);
        ::clReleaseMemObject(csrRowCInnProdNum);

        return (clsparseStatus)run_status;
    }

    // divide the rows into bins
    run_status = compute_nipBin(m, csrRowCInnProdNum, clinnBin, control);

    if (run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(clinnBin);
        ::clReleaseMemObject(csrRowCInnProdNum);

        return (clsparseStatus)run_status;
    }

    run_status = control->event.wait();

    if (run_status != CL_SUCCESS)
    {
        CLSPARSE_V(run_status, "control->event.wait()");
        ::clReleaseMemObject(clinnBin);
        ::clReleaseMemObject(csrRowCInnProdNum);

        return (clsparseStatus)run_status;
    }

    // retrieve the result to CPU
    int innBin[NIP_SEGMENTS];

    run_status = clEnqueueReadBuffer(control->queue(), clinnBin, CL_TRUE, 0, NIP_SEGMENTS * sizeof(int), 
                                     innBin, 0, NULL, NULL);

    if (run_status != CL_SUCCESS)
    {
        CLSPARSE_V(run_status, "clEnqueueReadBuffer(clinnBin, innBin)");
        ::clReleaseMemObject(clinnBin);
        ::clReleaseMemObject(csrRowCInnProdNum);

        return (clsparseStatus)run_status;
    }
    // free inner-product bins
    run_status = ::clReleaseMemObject(clinnBin);

    if (run_status != CL_SUCCESS)
    {
        CLSPARSE_V(run_status, "clReleaseMemObject(clinnBin)");
        ::clReleaseMemObject(csrRowCInnProdNum);

        return (clsparseStatus)run_status;
    }

    // print the statistics
    char *intprodstring[NIP_SEGMENTS] = {"0", "1", "2", "3~4", "5~8", "9~16", "17~32", "33~64", "65~128", 
                                         "129~256", "257~512", "513~1024", "1025~2048", "2049~4096", 
                                         "4097~8192", "8193~"};

    for (int i = 0; i < NIP_SEGMENTS; i++)
        printf("%s: %d\n", intprodstring[i], innBin[i]);

    // STAGE 3 Reorder rows of C based on the bins
    // set the base pointer of each bin based on the size of each bin
    int innPtr[NIP_SEGMENTS], i;

    innPtr[0] = 0;

    for (i = 1; i < NIP_SEGMENTS; i++)
        innPtr[i] = innPtr[i - 1] + innBin[i - 1];
    // copy the base pointers to GPU
    cl_mem clinnPtr = ::clCreateBuffer(cxt(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
                                       NIP_SEGMENTS * sizeof(cl_int), innPtr, &run_status);

    if (run_status != CL_SUCCESS)
    {
        CLSPARSE_V(run_status, "clinnPtr = clCreateBuffer()");
        ::clReleaseMemObject(csrRowCInnProdNum);

        return (clsparseStatus)run_status;
    }
    // buffer of reordered rows
    cl_mem csrRowCReorder = ::clCreateBuffer(cxt(), CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, 
                                             m * sizeof(cl_int), NULL, &run_status);

    if (run_status != CL_SUCCESS)
    {
        CLSPARSE_V(run_status, "csrRowCReorder = clCreateBuffer()");
        ::clReleaseMemObject(clinnPtr);
        ::clReleaseMemObject(csrRowCInnProdNum);

        return (clsparseStatus)run_status;
    }
    // reorder rows based on bins
    run_status = compute_reorderRowNip(m, csrRowCInnProdNum, clinnPtr, csrRowCReorder, control);

    if (run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(csrRowCReorder);
        ::clReleaseMemObject(clinnPtr);
        ::clReleaseMemObject(csrRowCInnProdNum);

        return (clsparseStatus)run_status;
    }

    run_status = control->event.wait();

    if (run_status != CL_SUCCESS)
    {
        CLSPARSE_V(run_status, "control->event.wait()");
        ::clReleaseMemObject(csrRowCReorder);
        ::clReleaseMemObject(clinnPtr);
        ::clReleaseMemObject(csrRowCInnProdNum);

        return (clsparseStatus)run_status;
    }
    // bin pointer will not be used further in GPU. free!
    run_status = ::clReleaseMemObject(clinnPtr);

    if (run_status != CL_SUCCESS)
    {
        CLSPARSE_V(run_status, "clReleaseMemObject(clinnPtr)");
        ::clReleaseMemObject(csrRowCReorder);
        ::clReleaseMemObject(csrRowCInnProdNum);

        return (clsparseStatus)run_status;
    }

    // STAGE 4  Count the number of non-zero elements of each row of output matrix for all groups
    // the number of non-zeros of each row of C.
    cl_mem csrRowCNnzSize = ::clCreateBuffer(cxt(), CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, 
                                             m * sizeof(cl_int), NULL, &run_status);

    if (run_status != CL_SUCCESS)
    {
        CLSPARSE_V(run_status, "csrRowCNnzSize = clCreateBuffer()");
        ::clReleaseMemObject(csrRowCReorder);
        ::clReleaseMemObject(csrRowCInnProdNum);

        return (clsparseStatus)run_status;
    }

    // maximun number of non-zeros in a row
    cl_mem clMaxNnz = ::clCreateBuffer(cxt(), CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, sizeof(cl_int), 
                                       NULL, &run_status);

    if (run_status != CL_SUCCESS)
    {
        CLSPARSE_V(run_status, "clMaxNnz = clCreateBuffer()");
        ::clReleaseMemObject(csrRowCNnzSize);
        ::clReleaseMemObject(csrRowCReorder);
        ::clReleaseMemObject(csrRowCInnProdNum);

        return (clsparseStatus)run_status;
    }

    cl_event nnz_event[2];
    // initialize as 0
    run_status = clEnqueueFillBuffer(control->queue(), csrRowCNnzSize, &pattern, sizeof(cl_int), 0, 
                                     m * sizeof(cl_int), 0, NULL, &nnz_event[0]);

    if (run_status != CL_SUCCESS)
    {
        CLSPARSE_V(run_status, "clEnqueueFillBuffer(csrRowCNnzSize)");
        ::clReleaseMemObject(clMaxNnz);
        ::clReleaseMemObject(csrRowCNnzSize);
        ::clReleaseMemObject(csrRowCReorder);
        ::clReleaseMemObject(csrRowCInnProdNum);

        return (clsparseStatus)run_status;
    }

    // initialize as 0
    run_status = clEnqueueFillBuffer(control->queue(), clMaxNnz, &pattern, sizeof(cl_int), 0, 
                                     sizeof(cl_int), 0, NULL, &nnz_event[1]);

    if (run_status != CL_SUCCESS)
    {
        CLSPARSE_V(run_status, "clEnqueueFillBuffer(clMaxNnz)");
        ::clReleaseMemObject(clMaxNnz);
        ::clReleaseMemObject(csrRowCNnzSize);
        ::clReleaseMemObject(csrRowCReorder);
        ::clReleaseMemObject(csrRowCInnProdNum);

        return (clsparseStatus)run_status;
    }

    run_status = ::clWaitForEvents(2, nnz_event);

    if (run_status != CL_SUCCESS)
    {
        CLSPARSE_V(run_status, "clWaitForEvents(nnz_event)");
        ::clReleaseMemObject(clMaxNnz);
        ::clReleaseMemObject(csrRowCNnzSize);
        ::clReleaseMemObject(csrRowCReorder);
        ::clReleaseMemObject(csrRowCInnProdNum);

        return (clsparseStatus)run_status;
    }

    run_status = ::clReleaseEvent(nnz_event[0]);

    if (run_status != CL_SUCCESS)
    {
        CLSPARSE_V(run_status, "clReleaseEvent(nnz_event[0]");
        ::clReleaseMemObject(clMaxNnz);
        ::clReleaseMemObject(csrRowCNnzSize);
        ::clReleaseMemObject(csrRowCReorder);
        ::clReleaseMemObject(csrRowCInnProdNum);

        return (clsparseStatus)run_status;
    }

    run_status = ::clReleaseEvent(nnz_event[1]);

    if (run_status != CL_SUCCESS)
    {
        CLSPARSE_V(run_status, "clReleaseEvent(nnz_event[1]");
        ::clReleaseMemObject(clMaxNnz);
        ::clReleaseMemObject(csrRowCNnzSize);
        ::clReleaseMemObject(csrRowCReorder);
        ::clReleaseMemObject(csrRowCInnProdNum);

        return (clsparseStatus)run_status;
    }

    // compute the number of non-zeros of each row of C
    run_status = compute_nnzC(csrRowPtrA, csrColIndA, csrRowPtrB, csrColIndB, csrRowCReorder, 
                              csrRowCNnzSize, innBin, innPtr, max_intprod, clMaxNnz, 
                              control);

    if (run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(clMaxNnz);
        ::clReleaseMemObject(csrRowCNnzSize);
        ::clReleaseMemObject(csrRowCReorder);
        ::clReleaseMemObject(csrRowCInnProdNum);

        return (clsparseStatus)run_status;
    }

    run_status = control->event.wait();

    if (run_status != CL_SUCCESS)
    {
        CLSPARSE_V(run_status, "control->event.wait()");
        ::clReleaseMemObject(clMaxNnz);
        ::clReleaseMemObject(csrRowCNnzSize);
        ::clReleaseMemObject(csrRowCReorder);
        ::clReleaseMemObject(csrRowCInnProdNum);

        return (clsparseStatus)run_status;
    }
    // the number of inner-product numbers are not be used again. free!
    run_status = ::clReleaseMemObject(csrRowCInnProdNum);

    if (run_status != CL_SUCCESS)
    {
        CLSPARSE_V(run_status, "clReleaseMemObject(csrRowCInnProdNum)");
        ::clReleaseMemObject(clMaxNnz);
        ::clReleaseMemObject(csrRowCReorder);
        ::clReleaseMemObject(csrRowCNnzSize);

        return (clsparseStatus)run_status;
    }
    // retrieve the maxinum number of non-zeros to CPU
    int max_nnz = 0;

    run_status = clEnqueueReadBuffer(control->queue(), clMaxNnz, CL_TRUE, 0, sizeof(int), &max_nnz, 0, 
                                     NULL, NULL);

    if (run_status != CL_SUCCESS)
    {
        CLSPARSE_V(run_status, "clEnqueueReadBuffer(clMaxNnz)");
        ::clReleaseMemObject(clMaxNnz);
        ::clReleaseMemObject(csrRowCReorder);
        ::clReleaseMemObject(csrRowCNnzSize);

        return (clsparseStatus)run_status;
    }
    // maximum number of non-zeros are not used later. free!
    run_status = ::clReleaseMemObject(clMaxNnz);

    if (run_status != CL_SUCCESS)
    {
        CLSPARSE_V(run_status, "clReleaseMemObject(clMaxNnz)");
        ::clReleaseMemObject(csrRowCReorder);
        ::clReleaseMemObject(csrRowCNnzSize);

        return (clsparseStatus)run_status;
    }

    // STAGE 5 Perform Prefix-sum for constructing C's row array
    matC->row_pointer = ::clCreateBuffer(cxt(), CL_MEM_READ_WRITE, (m + 1) * sizeof(cl_int), NULL, 
                                         &run_status);

    if (run_status != CL_SUCCESS)
    {;
        CLSPARSE_V(run_status, "matC->row_pointer = clCreateBuffer()")
        ::clReleaseMemObject(csrRowCReorder);
        ::clReleaseMemObject(csrRowCNnzSize);

        return (clsparseStatus)run_status;
    }

    cl_mem csrRowPtrC = matC->row_pointer;
    cl_event scan_event[2];
    // First element is 0
    run_status = clEnqueueFillBuffer(control->queue(), csrRowPtrC, &pattern, sizeof(cl_int), 0, 
                                     sizeof(cl_int), 0, NULL, &scan_event[0]);

    if (run_status != CL_SUCCESS)
    {
        CLSPARSE_V(run_status, "clEnqueueFillBuffer(csrRowPtrC)");
        ::clReleaseMemObject(csrRowPtrC);
        ::clReleaseMemObject(csrRowCReorder);
        ::clReleaseMemObject(csrRowCNnzSize);

        return (clsparseStatus)run_status;
    }
    // copy from csrRowCNnzSize to csrRowPtrC with 0~(m-1)'th elements to 1~m'th element
    run_status = clEnqueueCopyBuffer(control->queue(), csrRowCNnzSize, csrRowPtrC, 0, sizeof(cl_int), 
                                     m * sizeof(cl_int), 0, NULL, &scan_event[1]);

    if (run_status != CL_SUCCESS)
    {
        CLSPARSE_V(run_status, "clEnqueueCopyBuffer(csrRowCNnzSize, csrRowPtrC)");
        ::clReleaseMemObject(csrRowPtrC);
        ::clReleaseMemObject(csrRowCReorder);
        ::clReleaseMemObject(csrRowCNnzSize);

        return (clsparseStatus)run_status;
    }

    run_status = ::clWaitForEvents(2, scan_event);

    if (run_status != CL_SUCCESS)
    {
        CLSPARSE_V(run_status, "clWaitForEvents(scan_event)");
        ::clReleaseMemObject(csrRowPtrC);
        ::clReleaseMemObject(csrRowCReorder);
        ::clReleaseMemObject(csrRowCNnzSize);

        return (clsparseStatus)run_status;
    }

    run_status = ::clReleaseEvent(scan_event[0]);

    if (run_status != CL_SUCCESS)
    {
        CLSPARSE_V(run_status, "clReleaseEvent(scan_event[0]");
        ::clReleaseMemObject(csrRowPtrC);
        ::clReleaseMemObject(csrRowCReorder);
        ::clReleaseMemObject(csrRowCNnzSize);

        return (clsparseStatus)run_status;
    }

    run_status = ::clReleaseEvent(scan_event[1]);

    if (run_status != CL_SUCCESS)
    {
        CLSPARSE_V(run_status, "clReleaseEvent(scan_event[1]");
        ::clReleaseMemObject(csrRowPtrC);
        ::clReleaseMemObject(csrRowCReorder);
        ::clReleaseMemObject(csrRowCNnzSize);

        return (clsparseStatus)run_status;
    }
    // Perform prefix-scan on C's rows
    run_status = compute_scan(m + 1, csrRowPtrC, control);

    if (run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(csrRowPtrC);
        ::clReleaseMemObject(csrRowCReorder);
        ::clReleaseMemObject(csrRowCNnzSize);

        return (clsparseStatus)run_status;
    }

    // STAGE 6 Divide the rows into groups by the number of non-zero elements
    // bin: {0, 1, 2, 3~4, 5~8, 9~16, 17~32, 33~64, 65~128, 129~256, 257~512, 513~~1024, 1025~2048, 
    // 2049~4096, 4097~}
    cl_mem clNnzBin = ::clCreateBuffer(cxt(), CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, 
                                       NNZ_SEGMENTS * sizeof(cl_int), NULL, &run_status);

    if (run_status != CL_SUCCESS)
    {
        CLSPARSE_V(run_status, "clNnzBin = clCreateBuffer()");
        ::clReleaseMemObject(csrRowPtrC);
        ::clReleaseMemObject(csrRowCReorder);
        ::clReleaseMemObject(csrRowCNnzSize);

        return (clsparseStatus)run_status;
    }

    cl_event nnzbin_event;

    // initialize bins to 0
    run_status = clEnqueueFillBuffer(control->queue(), clNnzBin, &pattern, sizeof(cl_int), 0, 
                                     NNZ_SEGMENTS * sizeof(cl_int), 0, NULL, &nnzbin_event);

    if (run_status != CL_SUCCESS)
    {
        CLSPARSE_V(run_status, "clEnqueueFillBuffer(clNnzBin)");
        ::clReleaseMemObject(clNnzBin);
        ::clReleaseMemObject(csrRowPtrC);
        ::clReleaseMemObject(csrRowCReorder);
        ::clReleaseMemObject(csrRowCNnzSize);

        return (clsparseStatus)run_status;
    }

    run_status = ::clWaitForEvents(1, &nnzbin_event);

    if (run_status != CL_SUCCESS)
    {
        CLSPARSE_V(run_status, "clWaitForEvents(&nnzbin_event)");
        ::clReleaseMemObject(clNnzBin);
        ::clReleaseMemObject(csrRowPtrC);
        ::clReleaseMemObject(csrRowCReorder);
        ::clReleaseMemObject(csrRowCNnzSize);

        return (clsparseStatus)run_status;
    }

    run_status = ::clReleaseEvent(nnzbin_event);

    if (run_status != CL_SUCCESS)
    {
        CLSPARSE_V(run_status, "clReleaseEvent(nnzbin_event)");
        ::clReleaseMemObject(clNnzBin);
        ::clReleaseMemObject(csrRowPtrC);
        ::clReleaseMemObject(csrRowCReorder);
        ::clReleaseMemObject(csrRowCNnzSize);

        return (clsparseStatus)run_status;
    }

    run_status = compute_nnzBin(m, csrRowCNnzSize, clNnzBin, control);

    if (run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(clNnzBin);
        ::clReleaseMemObject(csrRowPtrC);
        ::clReleaseMemObject(csrRowCReorder);
        ::clReleaseMemObject(csrRowCNnzSize);

        return (clsparseStatus)run_status;
    }

    run_status = control->event.wait();

    if (run_status != CL_SUCCESS)
    {
        CLSPARSE_V(run_status, "control->event.wait()");
        ::clReleaseMemObject(clNnzBin);
        ::clReleaseMemObject(csrRowPtrC);
        ::clReleaseMemObject(csrRowCReorder);
        ::clReleaseMemObject(csrRowCNnzSize);

        return (clsparseStatus)run_status;
    }

    // retrieve bins to CPU
    int nnzBin[NNZ_SEGMENTS];

    run_status = clEnqueueReadBuffer(control->queue(), clNnzBin, CL_TRUE, 0, NNZ_SEGMENTS * sizeof(int),
                                     nnzBin, 0, NULL, NULL);

    if (run_status != CL_SUCCESS)
    {
        CLSPARSE_V(run_status, "clEnqueueReadBuffer(clNnzBin, nnzBin)");
        ::clReleaseMemObject(clNnzBin);
        ::clReleaseMemObject(csrRowPtrC);
        ::clReleaseMemObject(csrRowCNnzSize);

        return (clsparseStatus)run_status;
    }
    // bins are not used later on GPU. free!
    run_status = ::clReleaseMemObject(clNnzBin);

    if (run_status != CL_SUCCESS)
    {
        CLSPARSE_V(run_status, "clReleaseMemObject(clNnzBin)");
        ::clReleaseMemObject(csrRowPtrC);
        ::clReleaseMemObject(csrRowCReorder);
        ::clReleaseMemObject(csrRowCNnzSize);

        return (clsparseStatus)run_status;
    }
    // print the statistics
    char *nnzprodstring[NNZ_SEGMENTS] = {"0", "1", "2", "3~4", "5~8", "9~16", "17~32", "33~64", "65~128", 
                                         "129~256", "257~512", "513~1024", "1025~2048", "2049~4096", 
                                         "4097~"};

    for (int i = 0; i < NNZ_SEGMENTS; i++)
        printf("%s: %d\n", nnzprodstring[i], nnzBin[i]);

    // STAGE 7 Reorder rows of C based on the bins
    int nnzPtr[NNZ_SEGMENTS];

    nnzPtr[0] = 0;

    for (i = 1; i < NNZ_SEGMENTS; i++)
        nnzPtr[i] = nnzPtr[i - 1] + nnzBin[i - 1];
    // Set the base pointer buffer in GPU
    cl_mem clNnzPtr = ::clCreateBuffer(cxt(), CL_MEM_READ_WRITE | CL_MEM_HOST_WRITE_ONLY, 
                                       NNZ_SEGMENTS * sizeof(cl_int), NULL, &run_status);

    if (run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(csrRowPtrC);
        ::clReleaseMemObject(csrRowCReorder);
        ::clReleaseMemObject(csrRowCNnzSize);

        return (clsparseStatus)run_status;
    }

    cl_event reordernnz_event;
    // copy the base pointers to GPU
    run_status = clEnqueueWriteBuffer(control->queue(), clNnzPtr, CL_TRUE, 0, NNZ_SEGMENTS * sizeof(int), 
                                      nnzPtr, 0, NULL, &reordernnz_event);

    if (run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(clNnzPtr);
        ::clReleaseMemObject(csrRowPtrC);
        ::clReleaseMemObject(csrRowCReorder);
        ::clReleaseMemObject(csrRowCNnzSize);

        return (clsparseStatus)run_status;
    }

    run_status = ::clWaitForEvents(1, &reordernnz_event);

    if (run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(clNnzPtr);
        ::clReleaseMemObject(csrRowPtrC);
        ::clReleaseMemObject(csrRowCReorder);
        ::clReleaseMemObject(csrRowCNnzSize);

        return (clsparseStatus)run_status;
    }

    run_status = ::clReleaseEvent(reordernnz_event);

    if (run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(clNnzPtr);
        ::clReleaseMemObject(csrRowPtrC);
        ::clReleaseMemObject(csrRowCReorder);
        ::clReleaseMemObject(csrRowCNnzSize);

        return (clsparseStatus)run_status;
    }

    // reorder the rows based on bins
    run_status = compute_reorderRowNnz(m, csrRowCNnzSize, clNnzPtr, csrRowCReorder, control);

    if (run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(clNnzPtr);
        ::clReleaseMemObject(csrRowPtrC);
        ::clReleaseMemObject(csrRowCReorder);
        ::clReleaseMemObject(csrRowCNnzSize);

        return (clsparseStatus)run_status;
    }

    run_status = control->event.wait();

    if (run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(clNnzPtr);
        ::clReleaseMemObject(csrRowPtrC);
        ::clReleaseMemObject(csrRowCReorder);
        ::clReleaseMemObject(csrRowCNnzSize);

        return (clsparseStatus)run_status;
    }
    // pointer buffers are not used later. free!
    run_status = ::clReleaseMemObject(clNnzPtr);

    if (run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(csrRowPtrC);
        ::clReleaseMemObject(csrRowCReorder);
        ::clReleaseMemObject(csrRowCNnzSize);

        return (clsparseStatus)run_status;
    }

    run_status = ::clReleaseMemObject(csrRowCNnzSize);
    if(run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(csrRowPtrC);
        ::clReleaseMemObject(csrRowCReorder);

        return (clsparseStatus)run_status;
    }

    // STAGE 8 Compute values
    int nnzC;

    cl_event val_event;
    // read number of non-zeros of C to CPU
    run_status = clEnqueueReadBuffer(control->queue(), csrRowPtrC, CL_TRUE, m * sizeof(cl_int), 
                                     sizeof(cl_int), &nnzC, 0, NULL, val_event);

    if (run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(csrRowPtrC);
        ::clReleaseMemObject(csrRowCReorder);

        return (clsparseStatus)run_status;
    }
    // initialize column indices' array
    matC->col_indices = ::clCreateBuffer(cxt(), CL_MEM_READ_WRITE, nnzC * sizeof(cl_int), NULL, 
                                         &run_status);

    cl_mem csrColIndC = matC->col_indices;

    if (run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(csrRowPtrC);
        ::clReleaseMemObject(csrRowCReorder);

        return (clsparseStatus)run_status;
    }
    // initialize values' array
    matC->values = ::clCreateBuffer(cxt(), CL_MEM_READ_WRITE, nnzC * sizeof(cl_float), NULL, &run_status);

    cl_mem csrValC = matC->values;

    if (run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(csrRowPtrC);
        ::clReleaseMemObject(csrRowCReorder);

        return (clsparseStatus)run_status;
    }

    run_status = ::clWaitForEvents(1, &val_event);

    if (run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(csrRowPtrC);
        ::clReleaseMemObject(csrRowCReorder);

        return (clsparseStatus)run_status;
    }

    run_status = ::clReleaseEvent(val_event);

    if (run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(csrRowPtrC);
        ::clReleaseMemObject(csrRowCReorder);

        return (clsparseStatus)run_status;
    }

    // Compute the values
    run_status = compute_valC(csrRowPtrA, csrColIndA, csrValA, csrRowPtrB, csrColIndB, csrValB, 
                              csrRowPtrC, csrColIndC, csrValC, csrRowCReorder, nnzBin, nnzPtr, max_nnz, 
                              control);

    if (run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(csrValC);
        ::clReleaseMemObject(csrColIndC);
        ::clReleaseMemObject(csrRowPtrC);
        ::clReleaseMemObject(csrRowCReorder);

        return (clsparseStatus)run_status;
    }

    //Free buffers
    run_status = ::clReleaseMemObject(csrRowCReorder);

    if (run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(csrValC);
        ::clReleaseMemObject(csrColIndC);
        ::clReleaseMemObject(csrRowPtrC);

        return (clsparseStatus)run_status;
    }

    matC->num_rows = m;
    matC->num_cols = n;
    matC->num_nonzeros = nnzC;

    return clsparseSuccess;
}