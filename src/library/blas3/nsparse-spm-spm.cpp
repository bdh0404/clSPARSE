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
#include <stdio.h>

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

// compute inner-product numbers of each row, store the number in csrRowPtrinnSize, and the maximum number in clMaxIntProd
clsparseStatus compute_nnzCInnProdNum(int _m, cl_mem csrRowPtrA, cl_mem csrColIndA, cl_mem csrRowPtrB,
                                      cl_mem csrColIndB, cl_mem csrRowCInnProdNum, cl_mem clMaxIntProd, clsparseControl control)
{

    const std::string params = std::string() +
                               "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type;

    cl::Kernel kernel = KernelCache::get(control->queue, "SpGEMM_compute_InnProdNum_kernel", "compute_InnProdNum_kernel", params);

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    int num_threads = GROUPSIZE_256;
    size_t num_blocks = ceil((double)_m / (double)num_threads);

    szLocalWorkSize[0] = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    KernelWrap kWrapper(kernel);

    kWrapper << csrRowPtrA << csrColIndA << csrRowPtrB << csrRowCInnProdNum << cl::Local(256 * sizeof(int)) << clMaxIntProd << _m;

    cl::NDRange local(szLocalWorkSize[0]);
    cl::NDRange global(szGlobalWorkSize[0]);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

// count the size of the number of inner-products bins
clsparseStatus compute_nipBin(int _m, cl_mem csrRowCInnProdNum, cl_mem innBin, clsparseControl control)
{
    const std::string params = std::string() +
                               "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type;

    cl::Kernel kernel = KernelCache::get(control->queue, "SpGEMM_computeNipBin_kernels", "compute_nipBin_kernel", params);

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
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

// reorder the rows with bins and inner-product numbers
clsparseStatus compute_reorderRowNip(int _m, cl_mem csrRowCInnProdNum, cl_mem clinnPtr, cl_mem csrRowCReorder,
                                     clsparseControl control)
{
    const std::string params = std::string() +
                               "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type;

    cl::Kernel kernel = KernelCache::get(control->queue, "SpGEMM_computeReorderRowNip_kernels", "compute_ReorderRowNip_kernel", params);

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
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

// count the number of non-zero elements of rows with 0 inner-products
clsparseStatus compute_nnzC_0(cl_mem csrRowCReorder, cl_mem csrRowCNnzSize, int bin, clsparseControl control)
{
    const std::string params = std::string() +
                               "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type;

    cl::Kernel kernel = KernelCache::get(control->queue, "SpGEMM_computeNnz_kernels", "compute_nnzC_0", params);

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
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

// count the number of non-zero elements of rows with 1 inner-products
clsparseStatus compute_nnzC_1(cl_mem csrRowCReorder, cl_mem csrRowCNnzSize, int bin, int ptr,
                              clsparseControl control)
{
    const std::string params = std::string() +
                               "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type;

    cl::Kernel kernel = KernelCache::get(control->queue, "SpGEMM_computeNnz_kernels", "compute_nnzC_1", params);

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
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

// count the number of non-zero elements of rows with {2, 3~4, 5~8, 9~16, 17~32, 33~64} inner-products
clsparseStatus compute_nnzC_pwarp(cl_mem csrRowPtrA, cl_mem csrColIndA, cl_mem csrRowPtrB, cl_mem csrColIndB,
                                  cl_mem csrRowCReorder, cl_mem csrRowCNnzSize, int bin, int ptr, int innSize, clsparseControl control)
{
    const std::string params = std::string() +
                               "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type;

    cl::Kernel kernel = KernelCache::get(control->queue, "SpGEMM_computeNnz_kernels", "compute_nnzC_pwarp", params);

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    int num_threads = 64;
    size_t num_blocks = ceil((double)bin / (double)num_threads);

    szLocalWorkSize[0] = 64;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    KernelWrap kWrapper(kernel);
    kWrapper << csrRowPtrA << csrColIndA << csrRowPtrB << csrColIndB << csrRowCReorder << csrRowCNnzSize << cl::Local(64 * sizeof(int)) << bin << ptr << innSize;

    cl::NDRange local(szLocalWorkSize[0]);
    cl::NDRange global(szGlobalWorkSize[0]);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

// count the number of non-zero elements of rows with {65~128, 129~256, 257~512, 513~1024, 1025~2048, 2049~4096, 4097~8192} inner-products
clsparseStatus compute_nnzC_tb(int num_threads, cl_mem csrRowPtrA, cl_mem csrColIndA, cl_mem csrRowPtrB,
                               cl_mem csrColIndB, cl_mem csrRowCReorder, cl_mem csrRowCNnzSize, int bin, int ptr, int innSize,
                               clsparseControl control)
{
    const std::string params = std::string() +
                               "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type;

    cl::Kernel kernel = KernelCache::get(control->queue, "SpGEMM_computeNnz_kernels", "compute_nnzC_tb", params);

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    size_t num_blocks = ceil((double)bin / (double)num_threads);

    szLocalWorkSize[0] = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    KernelWrap kWrapper(kernel);
    kWrapper << csrRowPtrA << csrColIndA << csrRowPtrB << csrColIndB << csrRowCReorder << csrRowCNnzSize << cl::Local(innSize * sizeof(int)) << bin << ptr << innSize;

    cl::NDRange local(szLocalWorkSize[0]);
    cl::NDRange global(szGlobalWorkSize[0]);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

// attempt to count the number of non-zero elements of rows with 8193~ inner-products
clsparseStatus compute_nnzC_tb_large(cl_mem csrRowPtrA, cl_mem csrColIndA, cl_mem csrRowPtrB, cl_mem csrColIndB,
                                     cl_mem csrRowCReorder, cl_mem csrRowCNnzSize, cl_mem fail_count, cl_mem fail_perm, int bin, int ptr,
                                     clsparseControl control)
{
    const std::string params = std::string() +
                               "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type;

    cl::Kernel kernel = KernelCache::get(control->queue, "SpGEMM_computeNnz_kernels", "compute_nnzC_tb_large", params);

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    int num_threads = GROUPSIZE_256;
    size_t num_blocks = ceil((double)bin / (double)num_threads);

    szLocalWorkSize[0] = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    KernelWrap kWrapper(kernel);
    kWrapper << csrRowPtrA << csrColIndA << csrRowPtrB << csrColIndB << csrRowCReorder << csrRowCNnzSize << fail_count << fail_perm << cl::Local((MAX_HASH_SIZE - 1) * sizeof(int)) << cl::Local(sizeof(int)) << bin << ptr;

    cl::NDRange local(szLocalWorkSize[0]);
    cl::NDRange global(szGlobalWorkSize[0]);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

// count the number of non-zero elements of rows with 8193~ inner-products in global memory
clsparseStatus compute_nnzC_tb_global(cl_mem csrRowPtrA, cl_mem csrColIndA, cl_mem csrRowPtrB, cl_mem csrColIndB,
                                      cl_mem fail_perm, cl_mem csrRowCNnzSize, cl_mem clhashtable, int fail_count, int max_intprod, cl_mem clMaxNnz,
                                      clsparseControl control)
{
    const std::string params = std::string() +
                               "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type;

    cl::Kernel kernel = KernelCache::get(control->queue, "SpGEMM_computeNnz_kernels", "compute_nnzC_tb_large", params);

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    int num_threads = GROUPSIZE_256;
    size_t num_blocks = ceil((double)fail_count / (double)num_threads);

    szLocalWorkSize[0] = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    KernelWrap kWrapper(kernel);
    kWrapper << csrRowPtrA << csrColIndA << csrRowPtrB << csrColIndB << fail_perm << csrRowCNnzSize << clhashtable << fail_count << max_intprod;

    cl::NDRange local(szLocalWorkSize[0]);
    cl::NDRange global(szGlobalWorkSize[0]);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

// compute the number of non-zeros of each C's row based on inner-product number bins.
clsparseStatus compute_nnzC(cl_mem csrRowPtrA, cl_mem csrColIndA, cl_mem csrRowPtrB, cl_mem csrColIndB,
                            cl_mem csrRowCReorder, cl_mem csrRowCNnzSize, std::vector<int> innBin, std::vector<int> innPtr, int max_intprod, cl_mem clMaxNnz,
                            clsparseControl control)
{
    clsparseStatus run_status;

    if (innBin[0] > 0)
    {
        // rows with 0 intermediate products
        // 256 threads on each workgroup
        run_status = compute_nnzC_0(csrRowCReorder, csrRowCNnzSize, innBin[0], control);

        if(run_status != clsparseSuccess)
            return run_status;
    }
    if (innBin[1] > 0)
    {
        // rows with 1 intermediate product
        // 256 threads on each workgroup
        run_status = compute_nnzC_1(csrRowCReorder, csrRowCNnzSize, innBin[1], innPtr[1], control);

        if(run_status != clsparseSuccess)
            return run_status;
    }
        
    int i;
    for (i = 2; i <= 7; i++)
    {
        if (innBin[i] > 0)
        {
            // rows with {2, 3~4, 5~8, 9~16, 17~32, 33~64} intermediate products
            // 64 threads on each workgroup
            run_status = compute_nnzC_pwarp(csrRowPtrA, csrColIndA, csrRowPtrB, csrColIndB,
                               csrRowCReorder, csrRowCNnzSize, innBin[i], innPtr[i],
                               1 << (i - 1), control);
            
            if(run_status != clsparseSuccess)
                return run_status;
        }
    }
    for (i = 8; i < 12; i++)
    {
        if (innBin[i] > 0)
        {
            // rows with {65~128, 129~256, 257~512, 513~1024} intermediate products
            // 64 threads on each workgroup
            run_status = compute_nnzC_tb(64, csrRowPtrA, csrColIndA, csrRowPtrB,
                            csrColIndB, csrRowCReorder, csrRowCNnzSize, innBin[i], innPtr[i], control);
            
            if(run_status != clsparseSuccess)
                return run_status;
        }
    }

    for (i = 12; i < NUM_SEGMENTS - 1; i++)
    {
        if (innBin[i] > 0)
        {
            // rows with {1025~2048, 2049~4096, 4097~8192} intermediate products
            // 64, 128, 256 threads on each workgroup
            int num_threads = 1 << (i - 6);

            run_status = compute_nnzC_tb(num_threads, csrRowPtrA, csrColIndA, csrRowPtrB, csrColIndB, csrRowCReorder, csrRowCNnzSize, innBin[i], innPtr[i], control);

            if(run_status != clsparseSuccess)
                return run_status;
        }
    }
    if (innBin[15] > 0)
    {
        // rows with 8193~ intermediate products
        // 256 threads on each workgroup
        int pattern = 0;
        // count  for fail to fit in local memory
        cl_mem clfail_count = ::clCreateBuffer(cxt(), CL_MEM_HOST_READ_ONLY, sizeof(cl_int), NULL, &run_status);

        if(run_status != CL_SUCCESS)
            return run_status;
        // initialize to 0
        run_status = clEnqueueFillBuffer(control->queue(), clfail_count, &pattern, sizeof(cl_int), 0, sizeof(cl_int), 0, NULL, NULL);

        if(run_status != CL_SUCCESS)
        {
            ::clReleaseMemObject(clfail_count);

            return run_status;
        }
        // buffer that collects rows that fails to fit in local memory
        cl_mem clfail_perm = ::clCreateBuffer(cxt(), CL_MEM_HOST_NO_ACCESS, innBin[15] * sizeof(cl_int), NULL, &run_status);

        if(run_status != CL_SUCCESS)
        {
            ::clReleaseMemObject(clfail_count);

            return run_status;
        }
        // try to count the number of nonzeros of those large rows
        run_status = compute_nnzC_tb_large(csrRowPtrA, csrColIndA, csrRowPtrB, csrColIndB, csrRowCReorder, csrRowPtrCNnzSize, clfail_count, clfail_perm, innBin[15], ptr, control);

        if(run_status != CL_SUCCESS)
        {
            ::clReleaseMemObject(clfail_perm);
            ::clReleaseMemObject(clfail_count);

            return run_status;
        }
        // read the number of rows that failed to count in local memory in CPU
        int fail_count;

        run_status = clEnqueueReadBuffer(control->queue(),
                                         clfail_count,
                                         1,
                                         0,
                                         sizeof(int),
                                         &fail_count,
                                         0,
                                         0,
                                         0);
        
        if(run_status != CL_SUCCESS)
        {
            ::clReleaseMemObject(clfail_perm);
            ::clReleaseMemObject(clfail_count);

            return run_status;
        }
        // If there is failed rows:
        if (fail_count > 0)
        {
            // allocate hash table on global memory
            int minuspattern = -1;
            cl_mem clhashtable = ::clCreateBuffer(cxt(), CL_MEM_HOST_NO_ACCESS, max_intprod * fail_count * sizeof(cl_int), NULL, &run_status);

            if(run_status != CL_SUCCESS)
            {
                ::clReleaseMemObject(clfail_perm);
                ::clReleaseMemObject(clfail_count);

                return run_status;
            }
            // initialize hash table as -1
            run_status = clEnqueueFillBuffer(control->queue(), clfail_perm, &minuspattern, sizeof(cl_int), 0, max_intprod * fail_count * sizeof(cl_int), 0, NULL, NULL);

            if(run_status != CL_SUCCESS)
            {
                ::clReleaseMemObject(clhashtable);
                ::clReleaseMemObject(clfail_perm);
                ::clReleaseMemObject(clfail_count);

                return run_status;
            }
            // count the number of non-zeros on global memory
            run_status = compute_nnzC_global(csrRowPtrA, csrColIndA, csrRowPtrB, csrColIndB, clfail_perm, csrRowrCNnzSize, clhashtable, fail_count, clMaxNnz, control);

            if(run_status != CL_SUCCESS)
            {
                ::clReleaseMemObject(clhashtable);
                ::clReleaseMemObject(clfail_perm);
                ::clReleaseMemObject(clfail_count);

                return run_status;
            }
            // free hash table
            run_status = ::clReleaseMemObject(clhashtable);

            if(run_status != CL_SUCCESS)
            {
                ::clReleaseMemObject(clfail_perm);
                ::clReleaseMemObject(clfail_count);

                return run_status;
            }
        }
        // free failed row buffer
        run_status = ::clReleaseMemObject(clfail_perm);

        if(run_status != CL_SUCCESS)
        {
            ::clReleaseMemObject(clfail_count);

            return run_status;
        }
        // free failed row count
        run_status = ::clReleaseMemObject(clfail_count);

        if(run_status != CL_SUCCESS)
            return run_status;
    }

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

// perform prefix-scan to C's row pointer array
clsparseStatus compute_scan(int _m, cl_mem d_array, clsparseControl control)
{
    const std::string params = std::string() +
                               "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type;

    cl::Kernel scankernel = KernelCache::get(control->queue, "SpGEMM_scan_kernels", "scan_block", params);

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    int num_threads = GROUPSIZE_256;
    size_t num_blocks = ceil((double)_m / (double)(2 * GROUP_SIZE));

    szLocalWorkSize[0] = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    // allocate sum buffer that stores the last value of each block
    int pattern = 0;
    cl_int run_status;
    cl_mem d_sum = ::clCreateBuffer(cxt(), CL_MEM_HOST_NO_ACCESS, num_blocks * sizeof(cl_int), NULL, &run_status);

    if(run_status != CL_SUCCESS)
        return run_status;
    // perform prefix-scan in each block
    KernelWrap scankWrapper(scankernel);
    scankWrapper << d_array << cl::Local(2 * GROUPSIZE_256 * sizeof(int)) << d_sum << _m;

    cl::NDRange local(szLocalWorkSize[0]);
    cl::NDRange global(szGlobalWorkSize[0]);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        ::clReleaseMemObject(d_sum);

        return clsparseInvalidKernelExecution;
    }

    // if there are multiple sums
    if (_m > 1)
    {
        // scam the sum block
        clsparseStatus sparsestatus = compute_scan(num_blocks, d_sum, control);

        if (sparsestatus != clsparseSuccess)
        {
            ::clReleaseMemObject(d_sum);

            return clsparseInvalidKernelExecution;
        }
    }

    cl::Kernel addkernel = KernelCache::get(control->queue, "SpGEMM_scan_kernels", "add_block", params);
    // add the scanned sums into each block
    KernelWrap addkWrapper(addkernel);
    addkWrapper << d_array << d_sum << _m;

    cl_int status = addkWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        ::clReleaseMemObject(d_sum);

        return clsparseInvalidKernelExecution;
    }
    // release the sum buffer
    run_status = ::clReleaseMemObject(d_sum);

    if (run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(d_sum);

        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

// count the size of the number of non-zeros bins
clsparseStatus compute_nnzBin(int _m, cl_mem csrRowCNnzSize, cl_mem nnzBin, clsparseControl control)
{
    const std::string params = std::string() +
                               "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type;

    cl::Kernel kernel = KernelCache::get(control->queue, "SpGEMM_computeNipBin_kernels", "compute_nipBin_kernel", params);

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
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

// reorder the rows with bins and the number of non-zeros
clsparseStatus compute_reorderRowNnz(int _m, cl_mem csrRowCNnzSize, cl_mem clNnzPtr, cl_mem csrRowCReorder,
                                     clsparseControl control)
{
    const std::string params = std::string() +
                               "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type;

    cl::Kernel kernel = KernelCache::get(control->queue, "SpGEMM_computeReorderRowNip_kernels", "compute_ReorderRowNip_kernel", params);

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
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

/**
 * @brief 
 * 
 * @param csrRowPtrA 
 * @param csrColIndA 
 * @param csrColValA 
 * @param csrRowPtrB 
 * @param csrColIndB 
 * @param csrColValB 
 * @param csrRowPtrC 
 * @param csrColIndC 
 * @param csrColValC 
 * @param csrRowCReorder 
 * @param bin 
 * @param ptr 
 * @return clsparseStatus 
 */
clsparseStatus compute_valC_1(cl_mem csrRowPtrA, cl_mem csrColIndA, cl_mem csrColValA, cl_mem csrRowPtrB,
                              cl_mem csrColIndB, cl_mem csrColValB, cl_mem csrRowPtrC, cl_mem csrColIndC, cl_mem csrColValC, cl_mem csrRowCReorder,
                              int bin, int ptr, control)
{
    const std::string params = std::string() +
                               "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type;

    cl::Kernel kernel = KernelCache::get(control->queue, "SpGEMM_computeVal_kernels", "compute_valC_1", params);

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    int num_threads = GROUPSIZE_256;
    size_t num_blocks = ceil((double)bin / (double)num_threads);

    szLocalWorkSize[0] = GROUPSIZE_256;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    KernelWrap kWrapper(kernel);
    kWrapper << csrRowPtrA << csrColIndA << csrColValA << csrRowPtrB
             << csrColIndB << csrColValB << csrRowPtrC << csrColIndC
             << csrColValC << csrRowCReorder << bin << ptr;

    cl::NDRange local(szLocalWorkSize[0]);
    cl::NDRange global(szGlobalWorkSize[0]);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

// compute the value of rows with {2, 3~4, 5~8, 9~16, 17~32, 33~64} non-zero elements
clsparseStatus compute_valC_pwarp(cl_mem csrRowPtrA, cl_mem csrColIndA, cl_mem csrColValA, cl_mem csrRowPtrB,
                                  cl_mem csrColIndB, cl_mem csrColValB, cl_mem csrRowPtrC, cl_mem csrColIndC, cl_mem csrColValC, cl_mem csrRowCReorder,
                                  int bin, int ptr, int nnzSize, control)
{
    const std::string params = std::string() +
                               "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type;

    cl::Kernel kernel = KernelCache::get(control->queue, "SpGEMM_computeVal_kernels", "compute_valC_pwarp", params);

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    int num_threads = 64;
    size_t num_blocks = ceil((double)bin / (double)num_threads);

    szLocalWorkSize[0] = 64;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    KernelWrap kWrapper(kernel);
    kWrapper << csrRowPtrA << csrColIndA << csrColValA << csrRowPtrB
             << csrColIndB << csrColValB << csrRowPtrC << csrColIndC
             << csrColValC << csrRowCReorder << bin << ptr
             << nnzSize;

    cl::NDRange local(szLocalWorkSize[0]);
    cl::NDRange global(szGlobalWorkSize[0]);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

// compute the value of rows with {65~128, 129~256, 257~512, 513~1024, 1025~2048} non-zero elements
clsparseStatus compute_valC_tb(int num_threads, cl_mem csrRowPtrA, cl_mem csrColIndA, cl_mem csrColValA,
                               cl_mem csrRowPtrB, cl_mem csrColIndB, cl_mem csrColValB, cl_mem csrRowPtrC, cl_mem csrColIndC, cl_mem csrColValC,
                               cl_mem csrRowCReorder, int bin, int ptr, int nnzSize, control)
{
    const std::string params = std::string() +
                               "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type;

    cl::Kernel kernel = KernelCache::get(control->queue, "SpGEMM_computeVal_kernels", "compute_valC_tb", params);

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    size_t num_blocks = ceil((double)bin / (double)num_threads);

    szLocalWorkSize[0] = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    KernelWrap kWrapper(kernel);
    kWrapper << csrRowPtrA << csrColIndA << csrColValA << csrRowPtrB
             << csrColIndB << csrColValB << csrRowPtrC << csrColIndC
             << csrColValC << csrRowCReorder << bin << ptr
             << nnzSize;

    cl::NDRange local(szLocalWorkSize[0]);
    cl::NDRange global(szGlobalWorkSize[0]);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

// compute the value of rows with {2049~4096, 4096~} non-zero elements
clsparseStatus compute_valC_tb_global(int num_threads, cl_mem csrRowPtrA, cl_mem csrColIndA, cl_mem csrColValA,
                                      cl_mem csrRowPtrB, cl_mem csrColIndB, cl_mem csrColValB, cl_mem csrRowPtrC, cl_mem csrColIndC, cl_mem csrColValC,
                                      cl_mem csrRowCReorder, cl_mem clcolhashtable, cl_mem clvalhashtable, int bin, int ptr, int nnzSize,
                                      control)
{
    const std::string params = std::string() +
                               "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type;

    cl::Kernel kernel = KernelCache::get(control->queue, "SpGEMM_computeVal_kernels", "compute_valC_tb_global", params);

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    size_t num_blocks = ceil((double)bin / (double)num_threads);

    szLocalWorkSize[0] = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    KernelWrap kWrapper(kernel);
    kWrapper << csrRowPtrA << csrColIndA << csrColValA << csrRowPtrB
             << csrColIndB << csrColValB << csrRowPtrC << csrColIndC
             << csrColValC << csrRowCReorder << clcolhashtable << clvalhashtable
             << bin << ptr << nnzSize;

    cl::NDRange local(szLocalWorkSize[0]);
    cl::NDRange global(szGlobalWorkSize[0]);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

// compute the value of rows with non-zero elements
clsparseStatus compute_valC(cl_mem csrRowPtrA, cl_mem csrColIndA, cl_mem csrColValA, cl_mem csrRowPtrB,
                            cl_mem csrColIndB, cl_mem csrColValB, cl_mem csrRowPtrC, cl_mem csrColIndC, cl_mem csrColValC, cl_mem csrRowCReorder,
                            cl_mem csrRowCNnzSize, std::vector<int> nnzBin, std::vector<int> nnzPtr, int max_nnz, clsparseControl control)
{
    clsparseStatus run_status;
    if (nnzBin[1] > 0)
    {
        // compute values with 1 non-zero element
        run_status = compute_valC_1(csrRowPtrA, csrColIndA, csrColValA, csrRowPtrB,
                       csrColIndB, csrColValB, csrRowPtrC, csrColIndC,
                       csrColValC, csrRowCReorder, nnzBin[1], nnzPtr[1],
                       control);

        if(run_status != clsparseSuccess)
            return run_status;
    }

    int i;

    for (i = 2; i <= 7; i++)
    {
        if (nnzBin[i] > 0)
        {
            // compute values with {2, 3~4, 5~8, 9~16, 17~32, 33~64} non-zero elements
            run_status = compute_valC_pwarp(csrRowPtrA, csrColIndA, csrColValA, csrRowPtrB,
                               csrColIndB, csrColValB, csrRowPtrC, csrColIndC,
                               csrColValC, csrRowCReorder, csrRowCNnzSize, nnzBin[i],
                               nnzPtr[i], 1 << (i - 1), control);
            if(run_status != clsparseSuccess)
                return run_status;
        }
    }
    for (i = 8; i < 10; i++)
    {
        if (nnzBin[i] > 0)
        {
            // compute values with {65~128, 129~256} non-zero elements
            run_status = compute_valC_tb(64, csrRowPtrA, csrColIndA, csrColValA,
                            csrRowPtrB, csrColIndB, csrColValB, csrRowPtrC,
                            csrColIndC, csrColValC, csrRowCReorder, csrRowCNnzSize,
                            nnzBin[i], nnzPtr[i], 1 << (i - 1), control);
            if(run_status != clsparseSuccess)
                return run_status;
        }
    }

    for (i = 10; i < 12; i++)
    {
        if (nnzBin[i] > 0)
        {
            // compute values with {257~512, 513~1024, 1025~2048} non-zero elements
            int num_threads = 1 << (i - 6);

            run_status = compute_valC_tb(num_threads, csrRowPtrA, csrColIndA, csrColValA,
                            csrRowPtrB, csrColIndB, csrColValB, csrRowPtrC,
                            csrColIndC, csrColValC, csrRowCReorder, csrRowCNnzSize,
                            nnzBin[i], nnzPtr[i], 1 << (i - 1), control);

            if(run_status != clsparseSuccess)
                return run_status;
        }
    }
    if (nnzBin[13] > 0)
    {
        // compute values with 2049~4096 non-zero elements
        int pattern = 0;
        int minuspattern = -1;
        // allocate column hash table with size 8192 * sizeof(cl_int)
        cl_mem clcolhashtable = ::clCreateBuffer(cxt(), CL_MEM_HOST_NO_ACCESS, 8192 * sizeof(cl_int), NULL, &run_status);

        if(run_status != CL_SUCCESS)
            return run_status;
        // initialize column hash table as -1
        run_status = clEnqueueFillBuffer(control->queue(), clcolhashtable, &minuspattern, sizeof(cl_int), 0, nnzBin[13] * sizeof(cl_int), 0, NULL, NULL);

        if(run_status != CL_SUCCESS)
        {
            ::clReleaseMemObject(clcolhashtable);

            return run_status;
        }
        // allocate value hash table with size 8192 * sizeof(cl_int)
        cl_mem clvalhashtable = ::clCreateBuffer(cxt(), CL_MEM_HOST_NO_ACCESS, 8192 * sizeof(cl_int), NULL, &run_status);

        if(run_status != CL_SUCCESS)
        {
            ::clReleaseMemObject(clcolhashtable);
            
            return run_status;
        }
        // initialize value hash table as 0
        run_status = clEnqueueFillBuffer(control->queue(), clvalhashtable, &pattern, sizeof(cl_int), 0, nnzBin[13] * sizeof(cl_int), 0, NULL, NULL);

        if(run_status != CL_SUCCESS)
        {
            ::clReleaseMemObject(clvalhashtable);
            ::clReleaseMemObject(clcolhashtable);
            
            return run_status;
        }
        // compute values on global hash table
        run_status = compute_valC_global(num_threads, csrRowPtrA, csrColIndA, csrColValA,
                            csrRowPtrB, csrColIndB, csrColValB, csrRowPtrC,
                            csrColIndC, csrColValC, csrRowCReorder, csrRowCNnzSize,
                            clcolhashtable, clvalhashtable, nnzBin[i], nnzPtr[i],
                            1 << (i - 1), control);

        if(run_status != CL_SUCCESS)
        {
            ::clReleaseMemObject(clvalhashtable);
            ::clReleaseMemObject(clcolhashtable);
            
            return run_status;
        }
        // free value hash table
        run_status = ::clReleaseMemObject(clvalhashtable);

        if(run_status != CL_SUCCESS)
        {
            ::clReleaseMemObject(clcolhashtable);
            
            return run_status;
        }
        // free column hash table
        run_status = ::clReleaseMemObject(clcolhashtable);

        if(run_status != CL_SUCCESS)
            return run_status;
    }
    if (innBin[14] > 0)
    {
        // compute values with 4097~ non-zero elements
        int pattern = 0, minuspattern = -1;
        // allocate column hash table with size 8192 * sizeof(cl_int)
        cl_mem clcolhashtable = ::clCreateBuffer(cxt(), CL_MEM_HOST_NO_ACCESS, 8192 * sizeof(cl_int), NULL, &run_status);

        if(run_status != CL_SUCCESS)
            return run_status;
        // initialize column hash table as -1
        run_status = clEnqueueFillBuffer(control->queue(), clcolhashtable, &minuspattern, sizeof(cl_int), 0, 2 * max_nnz * sizeof(cl_int), 0, NULL, NULL);

        if(run_status != CL_SUCCESS)
        {
            ::clReleaseMemObject(clcolhashtable);

            return run_status;
        }
        // allocate value hash table with size 8192 * sizeof(cl_int)
        cl_mem clvalhashtable = ::clCreateBuffer(cxt(), CL_MEM_HOST_NO_ACCESS, 8192 * sizeof(cl_int), NULL, &run_status);

        if(run_status != CL_SUCCESS)
        {
            ::clReleaseMemObject(clcolhashtable);

            return run_status;
        }
        // initialize value hash table as 0
        run_status = clEnqueueFillBuffer(control->queue(), clvalhashtable, &pattern, sizeof(cl_int), 0, 2 * max_nnz * sizeof(cl_int), 0, NULL, NULL);

        if(run_status != CL_SUCCESS)
        {
            ::clReleaseMemObject(clvalhashtable);
            ::clReleaseMemObject(clcolhashtable);

            return run_status;
        }
        // compute values on global hash table
        run_status = compute_valC_global(num_threads, csrRowPtrA, csrColIndA, csrColValA,
                            csrRowPtrB, csrColIndB, csrColValB, csrRowPtrC,
                            csrColIndC, csrColValC, csrRowCReorder, csrRowCNnzSize,
                            clcolhashtable, clvalhashtable, nnzBin[i], nnzPtr[i],
                            max_nnz, control);
        
        if(run_status != CL_SUCCESS)
        {
            ::clReleaseMemObject(clvalhashtable);
            ::clReleaseMemObject(clcolhashtable);

            return run_status;
        }
        // free value hash table
        run_status = ::clReleaseMemObject(clvalhashtable);

        if(run_status != CL_SUCCESS)
        {
            ::clReleaseMemObject(clcolhashtable);

            return run_status;
        }
        // free column hash table
        run_status = ::clReleaseMemObject(clcolhashtable);

        if(run_status != CL_SUCCESS)
            return run_status;
    }

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

// C = A * B, A, B, C are CSR Matrix
CLSPARSE_EXPORT clsparseStatus
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
    cl_mem csrColValA = matA->values;
    cl_mem csrRowPtrB = matB->row_pointer;
    cl_mem csrColIndB = matB->col_indices;
    cl_mem csrColValB = matB->values;

    cl::Context cxt = control->getContext();

    // STAGE 1 Count the number of intermediate products of each row
    int pattern = 0;

    // inner-product numbers of each C's row, m integers.
    cl_mem csrRowCInnProdNum = ::clCreateBuffer(cxt(), CL_MEM_HOST_NO_ACCESS, m * sizeof(cl_int), NULL, &run_status);

    if(run_status != CL_SUCCESS)
        return run_status;
    // Fill csrRowInnProdNum as 0
    run_status = clEnqueueFillBuffer(control->queue(), csrRowPtrCInnProdNum, &pattern, sizeof(cl_int), 0, m * sizeof(cl_int), 0, NULL, NULL);

    if(run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(csrRowInnProdNum);

        return run_status;
    }
    // maximum inner-product number of all rows
    cl_mem clMaxIntProd = ::clCreateBuffer(cxt(), CL_MEM_HOST_READ_ONLY, sizeof(cl_int), NULL, &run_status);

    if(run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(csrRowInnProdNum);

        return run_status;
    }
    // initialize to 0
    run_status = clEnqueueFillBuffer(control->queue(), clMaxIntProd, &pattern, sizeof(cl_int), 0, sizeof(cl_int), 0, NULL, NULL);

    if(run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(clMaxIntProd);
        ::clReleaseMemObject(csrRowInnProdNum);

        return run_status;
    }
    // calculate the numbers of inner-product
    run_status = compute_nnzCInnProdNum(m, csrRowPtrA, csrColIndA, csrRowPtrB, csrColIndB, csrRowCInnProdNum, clMaxIntProd, control);

    if(run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(clMaxIntProd);
        ::clReleaseMemObject(csrRowInnProdNum);

        return run_status;
    }

    int max_intprod;
    // retrieve the maximum inner-product number to CPU.
    run_status = clEnqueueReadBuffer(control->queue(),
                                     clMaxIntProd,
                                     1,
                                     0,
                                     sizeof(int),
                                     &max_intprod,
                                     0,
                                     0,
                                     0);
    
    if(run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(clMaxIntProd);
        ::clReleaseMemObject(csrRowInnProdNum);

        return run_status;
    }
    // clMaxIntProd will not be used again: free!
    run_status = ::clReleaseMemObject(clMaxIntProd);

    if(run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(csrRowInnProdNum);

        return run_status;
    }

    // STAGE 2 Divide the rows into groups by the number of intermediate products
    // bins of rows with inner-product numbers
    // local memory size: 32KB.
    // 16 members of bin: {0, 1, 2, 3~4, 5~8, 9~16, 17~32, 33~64, 65~128, 129~256, 257~512, 513~1024, 1025~2048, 2049~4096, 4097~8192, 8192~}
    cl_mem clinnBin = ::clCreateBuffer(cxt(), CL_MEM_HOST_READ_ONLY, NIP_SEGMENTS * sizeof(cl_int), NULL, &run_status);

    if(run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(csrRowInnProdNum);

        return run_status;
    }

    run_status = clEnqueueFillBuffer(control->queue(), clinnBin, &pattern, sizeof(cl_int), 0, NIP_SEGMENTS * sizeof(cl_int), 0, NULL, NULL);

    if(run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(clinnBin);
        ::clReleaseMemObject(csrRowInnProdNum);

        return run_status;
    }

    // divide the rows into bins
    run_status = compute_nipBin(m, csrRowCInnProdNum, clinnBin, control);

    if(run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(clinnBin);
        ::clReleaseMemObject(csrRowInnProdNum);

        return run_status;
    }
    // retrieve the result to CPU
    int innBin[NIP_SEGMENTS];

    run_status = clEnqueueReadBuffer(control->queue(),
                                     clinnBin,
                                     1,
                                     0,
                                     NIP_SEGMENTS * sizeof(int),
                                     innBin,
                                     0,
                                     0,
                                     0);
    
    if(run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(clinnBin);
        ::clReleaseMemObject(csrRowInnProdNum);
        
        return run_status;
    }
    // free inner-product bins
    run_status = ::clReleaseMemObject(clinnBin);

    if(run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(csrRowInnProdNum);
        
        return run_status;
    }

    // print the statistics
    char *intprodstring[NIP_SEGMENTS] = {"0", "1", "2", "3~4", "5~8", "9~16", "17~32", "33~64", "65~128", "129~256", "257~512", "513~1024", "1025~2048", "2049~4096", "4097~8192", "8193~"};

    for (int i = 0; i < NIP_SEGMENTS; i++)
        printf("%s: %d\n", intprodstring[i], innBin[i]);

    // STAGE 3 Reorder rows of C based on the bins
    // set the base pointer of each bin based on the size of each bin
    int innPtr[NIP_SEGMENTS], i;

    innPtr[0] = 0;

    for (i = 1; i < NIP_SEGMENTS; i++)
        innPtr[i] = innPtr[i - 1] + innBin[i - 1];
    // copy the base pointers to GPU
    cl_mem clinnPtr = ::clCreateBuffer(cxt(), CL_MEM_COPY_HOST_PTR, NIP_SEGMENTS * sizeof(cl_int), innPtr, &run_status);

    if(run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(csrRowInnProdNum);
        
        return run_status;
    }
    // buffer of reordered rows
    cl_mem csrRowCReorder = ::clCreateBuffer(cxt(), CL_MEM_HOST_NO_ACCESS, m * sizeof(cl_int), NULL, &run_status);

    if(run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(clinnPtr);
        ::clReleaseMemObject(csrRowInnProdNum);
        
        return run_status;
    }
    // reorder rows based on bins
    run_status = compute_reorderRowNip(m, csrRowCInnProdNum, clinnPtr, csrRowCReorder, control);

    if(run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(clinnPtr);
        ::clReleaseMemObject(csrRowInnProdNum);
        
        return run_status;
    }
    // bin pointer will not be used further in GPU. free!
    run_status = ::clReleaseMemObject(clinnPtr);

    if(run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(csrRowInnProdNum);
        
        return run_status;
    }

    // STAGE 4  Count the number of non-zero elements of each row of output matrix for all groups
    // the number of non-zeros of each row of C.
    cl_mem csrRowCNnzSize = ::clCreateBuffer(cxt(), CL_MEM_HOST_NO_ACCESS, m * sizeof(cl_int), NULL, &run_status);

    if(run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(csrRowInnProdNum);
        
        return run_status;
    }
    // initialize as 0
    run_status = clEnqueueFillBuffer(control->queue(), csrRowCNnzSize, &pattern, sizeof(cl_int), 0, m * sizeof(cl_int), 0, NULL, NULL);

    if(run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(csrRowCNnzSize);
        ::clReleaseMemObject(csrRowInnProdNum);
        
        return run_status;
    }
    // maximun number of non-zeros in a row
    cl_mem clMaxNnz = ::clCreateBuffer(cxt(), CL_MEM_HOST_READ_ONLY, sizeof(cl_int), NULL, &run_status);

    if(run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(csrRowCNnzSize);
        ::clReleaseMemObject(csrRowInnProdNum);
        
        return run_status;
    }
    // initialize as 0
    run_status = clEnqueueFillBuffer(control->queue(), clMaxNnz, &pattern, sizeof(cl_int), 0, sizeof(cl_int), 0, NULL, NULL);

    if(run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(clMaxNnz);
        ::clReleaseMemObject(csrRowCNnzSize);
        ::clReleaseMemObject(csrRowInnProdNum);
        
        return run_status;
    }
    // compute the number of non-zeros of each row of C
    run_status = compute_nnzC(csrRowPtrA, csrColIndA, csrRowPtrB, csrColIndB, csrRowCReorder, csrRowCInnProdNum, csrRowCNnzSize, innBin, innPtr, max_intprod, clMaxNnz, control);

    if(run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(clMaxNnz);
        ::clReleaseMemObject(csrRowCNnzSize);
        ::clReleaseMemObject(csrRowInnProdNum);
        
        return run_status;
    }
    // the number of inner-product numbers are not be used again. free!
    run_status = ::clReleaseMemObject(csrRowCInnProdNum);

    if(run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(clMaxNnz);
        ::clReleaseMemObject(csrRowCNnzSize);
        
        return run_status;
    }
    // retrieve the maxinum number of non-zeros to CPU
    int max_nnz = 0;

    run_status = clEnqueueReadBuffer(control->queue(),
                                     clMaxNnz,
                                     1,
                                     0,
                                     sizeof(int),
                                     &max_nnz,
                                     0,
                                     0,
                                     0);
    
    if(run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(clMaxNnz);
        ::clReleaseMemObject(csrRowCNnzSize);
        
        return run_status;
    }
    // maximum number of non-zeros are not used later. free!
    run_status = ::clReleaseMemObject(clMaxNnz);

    if(run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(csrRowCNnzSize);
        
        return run_status;
    }

    // STAGE 5 Perform Prefix-sum for constructing C's row array
    matC->row_pointer = ::clCreateBuffer(cxt(), CL_MEM_READ_WRITE, (m + 1) * sizeof(cl_int), NULL, &run_status);

    if(run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(csrRowCNnzSize);
        
        return run_status;
    }

    cl_mem csrRowPtrC = matC->row_pointer;
    // First element is 0
    run_status = clEnqueueFillBuffer(control->queue(), csrRowPtrC, &pattern, sizeof(cl_int), 0, sizeof(cl_int), 0, NULL, NULL);

    if(run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(csrRowPtrC);
        ::clReleaseMemObject(csrRowCNnzSize);
        
        return run_status;
    }
    // copy from csrRowCNnzSize to csrRowPtrC with 0~(m-1)'th elements to 1~m'th element
    run_status = clEnqueueCopyBuffer(control->queue(), csrRowCNnzSize, csrRowPtrC, 0, sizeof(cl_int), m * sizeof(cl_int), 0, NULL, NULL);

    if(run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(csrRowPtrC);
        ::clReleaseMemObject(csrRowCNnzSize);
        
        return run_status;
    }
    // Perform prefix-scan on C's rows
    run_status = compute_scan(m + 1, csrRowPtrC, control);

    if(run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(csrRowPtrC);
        ::clReleaseMemObject(csrRowCNnzSize);
        
        return run_status;
    }

    // STAGE 6 Divide the rows into groups by the number of non-zero elements
    // bin: {0, 1, 2, 3~4, 5~8, 9~16, 17~32, 33~64, 65~128, 129~256, 257~512, 513~~1024, 1025~2048, 2049~4096, 4097~}
    cl_mem clNnzBin = ::clCreateBuffer(cxt(), CL_MEM_HOST_READ_ONLY, NNZ_SEGMENTS * sizeof(cl_int), NULL, &run_status);

    if(run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(csrRowPtrC);
        ::clReleaseMemObject(csrRowCNnzSize);
        
        return run_status;
    }
    // initialize bins to 0
    run_status = clEnqueueFillBuffer(control->queue(), clNnzBin, &pattern, sizeof(cl_int), 0, NNZ_SEGMENTS * sizeof(cl_int), 0, NULL, NULL);

    if(run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(clNnzBin);
        ::clReleaseMemObject(csrRowPtrC);
        ::clReleaseMemObject(csrRowCNnzSize);
        
        return run_status;
    }

    run_status = compute_nnzBin(m, csrRowCNnzSize, clNnzBin, control);

    if(run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(clNnzBin);
        ::clReleaseMemObject(csrRowPtrC);
        ::clReleaseMemObject(csrRowCNnzSize);
        
        return run_status;
    }
    // retrieve bins to CPU
    int nnzBin[NNZ_SEGMENTS];

    run_status = clEnqueueReadBuffer(control->queue(),
                                     clNnzBin,
                                     1,
                                     0,
                                     NNZ_SEGMENTS * sizeof(int),
                                     nnzBin,
                                     0,
                                     0,
                                     0);
    
    if(run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(clNnzBin);
        ::clReleaseMemObject(csrRowPtrC);
        ::clReleaseMemObject(csrRowCNnzSize);
        
        return run_status;
    }
    // bins are not used later on GPU. free!
    run_status = ::clReleaseMemObject(clNnzBin);
    
    if(run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(csrRowPtrC);
        ::clReleaseMemObject(csrRowCNnzSize);
        
        return run_status;
    }
    // print the statistics
    char *nnzprodstring[NNZ_SEGMENTS] = {"0", "1", "2", "3~4", "5~8", "9~16", "17~32", "33~64", "65~128", "129~256", "257~512", "513~1024", "1025~2048", "2049~4096", "4097~"};

    for (int i = 0; i < NNZ_SEGMENTS; i++)
        printf("%s: %d\n", nnzprodstring[i], nnzBin[i]);

    // STAGE 7 Reorder rows of C based on the bins
    int nnzPtr[NNZ_SEGMENTS];

    nnzPtr[0] = 0;

    for (i = 1; i < NNZ_SEGMENTS; i++)
        nnzPtr[i] = nnzPtr[i - 1] + nnzBin[i - 1];
    // Set the base pointer buffer in GPU
    cl_mem clNnzPtr = ::clCreateBuffer(cxt(), CL_MEM_HOST_WRITE_ONLY, NNZ_SEGMENTS * sizeof(cl_int), NULL, &run_status);

    if(run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(csrRowPtrC);
        ::clReleaseMemObject(csrRowCNnzSize);
        
        return run_status;
    }
    // copy the base pointers to GPU
    run_status = clEnqueueWriteBuffer(control->queue(),
                                      clNnzPtr,
                                      1,
                                      0,
                                      NNZ_SEGMENTS * sizeof(int),
                                      nnzPtr,
                                      0,
                                      0,
                                      0);

    if(run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(clNnzPtr);
        ::clReleaseMemObject(csrRowPtrC);
        ::clReleaseMemObject(csrRowCNnzSize);
        
        return run_status;
    }
    // reorder the rows based on bins
    run_status = compute_reorderRowNnz(m, csrRowCNnzSize, clNnzPtr, csrRowCReorder, control);

    if(run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(clNnzPtr);
        ::clReleaseMemObject(csrRowPtrC);
        ::clReleaseMemObject(csrRowCNnzSize);
        
        return run_status;
    }
    // pointer buffers are not used later. free!
    run_status = ::clReleaseMemObject(clNnzPtr);

    if(run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(csrRowPtrC);
        ::clReleaseMemObject(csrRowCNnzSize);
        
        return run_status;
    }

    // STAGE 8 Compute values
    int nnzC;
    // read number of non-zeros of C to CPU
    run_status = clEnqueueReadBuffer(control->queue(), csrRowPtrC, 1, m * sizeof(cl_int), sizeof(cl_int), &nnzC, 0, 0, 0);

    if(run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(csrRowPtrC);
        ::clReleaseMemObject(csrRowCNnzSize);
        
        return run_status;
    }
    // initialize column indices' array
    matC->col_indices = ::clCreateBuffer(cxt(), CL_MEM_READ_WRITE, nnzC * sizeof(cl_int), NULL, &run_status);

    cl_mem csrColIndC = matC->col_indices;

    if(run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(csrRowPtrC);
        ::clReleaseMemObject(csrRowCNnzSize);
        
        return run_status;
    }
    // initialize values' array
    matC->values = ::clCreateBuffer(cxt(), CL_MEM_READ_WRITE, nnzC * sizeof(cl_float), NULL, &run_status);

    cl_mem csrColValC = matC->values;

    if(run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(csrColIndC);
        ::clReleaseMemObject(csrRowPtrC);
        ::clReleaseMemObject(csrRowCNnzSize);
        
        return run_status;
    }

    // Compute the values
    run_status = compute_valC(csrRowPtrA, csrColIndA, csrColValA, csrRowPtrB,
                 csrColIndB, csrColValB, csrRowPtrC, csrColIndC,
                 csrColValC, csrRowCReorder, csrRowCNnzSize, nnzBin,
                 nnzPtr, max_nnz, control);

    if(run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(csrColIndC);
        ::clReleaseMemObject(csrRowPtrC);
        ::clReleaseMemObject(csrRowCNnzSize);
        
        return run_status;
    }

    //Free buffers
    run_status = ::clReleaseMemObject(csrRowCReorder);

    if(run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(csrColValC);
        ::clReleaseMemObject(csrColIndC);
        ::clReleaseMemObject(csrRowPtrC);
        ::clReleaseMemObject(csrRowCNnzSize);
        
        return run_status;
    }

    run_status = ::clReleaseMemObject(csrRowCNnzSize);

    if(run_status != CL_SUCCESS)
    {
        ::clReleaseMemObject(csrColValC);
        ::clReleaseMemObject(csrColIndC);
        ::clReleaseMemObject(csrRowPtrC);
        ::clReleaseMemObject(csrRowCNnzSize);
        
        return run_status;
    }

    matC->num_rows = m;
    matC->num_cols = n;
    matC->num_nonzeros = nnzC;

    return clsparseSuccess;
}