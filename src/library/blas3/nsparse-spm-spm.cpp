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

#include <cmath>

#define GROUPSIZE_256 256
#define TUPLE_QUEUE 6
#define INT_PROD_NUM_SEGMENTS 16
//#define WARPSIZE_NV_2HEAP 64
#define value_type float
#define index_type int
#define NSPARSE_SUCCESS 0
#define MAX_HASH_SIZE 8192

using namespace std;

int statistics(int *_h_csrRowPtrCt, int *_h_counter, int *_h_counter_one, int *_h_counter_sum, int *_h_queue_one, int _m);

// compute interproduct numbers of each row, store the number in csrRowPtrIntSize, and the maximum number in clMaxIntProd
clsparseStatus compute_nnzCIntProdNum(int _m, cl_mem csrRowPtrA, cl_mem csrColIndA, cl_mem csrRowPtrB, cl_mem csrColIndB, cl_mem csrRowCIntProdNum, cl_mem clMaxIntProd, clsparseControl control){

     const std::string params = std::string() +
               "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type
            + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type;

    cl::Kernel kernel = KernelCache::get(control->queue,"SpGEMM_compute_IntProdNum_kernel", "compute_IntProdNum_kernel", params);

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    int num_threads = GROUPSIZE_256;
    size_t num_blocks = ceil((double)_m / (double)num_threads);

    szLocalWorkSize[0]  = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    KernelWrap kWrapper(kernel);

    kWrapper << csrRowPtrA << csrColIndA << csrRowPtrB << csrRowCIntProdNum << cl::Local(256 * sizeof(int)) << clMaxIntProd << _m;

    cl::NDRange local(szLocalWorkSize[0]);
    cl::NDRange global(szGlobalWorkSize[0]);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
 }

clsparseStatus compute_nnzIntBin(int _m, cl_mem csrRowCIntProdNum, cl_mem intBin, clsparseControl control)
{
     const std::string params = std::string() +
               "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type
            + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type;

    cl::Kernel kernel = KernelCache::get(control->queue,"SpGEMM_computeNnzIIntBin_kernels", "compute_nnzIntBin_kernel", params);

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    int num_threads = GROUPSIZE_256;
    size_t num_blocks = ceil((double)_m / (double)num_threads);

    szLocalWorkSize[0]  = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    KernelWrap kWrapper(kernel);

    kWrapper << csrRowCIntProdNum << cl::Local(INT_PROD_NUM_SEGMENTS * sizeof(int) ) << intBin << _m;

    cl::NDRange local(szLocalWorkSize[0]);
    cl::NDRange global(szGlobalWorkSize[0]);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

clsparseStatus compute_reorderRow(int _m, cl_mem csrRowCIntProdNum, cl_mem clIntPtr, cl_mem csrRowCReorder, clsparseControl control)
{
    const std::string params = std::string() +
               "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type
            + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type;

    cl::Kernel kernel = KernelCache::get(control->queue,"SpGEMM_computeReorderRow_kernels", "compute_ReorderRow_kernel", params);

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    int num_threads = GROUPSIZE_256;
    size_t num_blocks = ceil((double)_m / (double)num_threads);

    szLocalWorkSize[0]  = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    KernelWrap kWrapper(kernel);

    kWrapper << csrRowCIntProdNum << clIntPtr << csrRowCReorder << _m;

    cl::NDRange local(szLocalWorkSize[0]);
    cl::NDRange global(szGlobalWorkSize[0]);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

clsparseStatus compute_nnzC(cl_mem csrRowPtrA, cl_mem csrColIndA, cl_mem csrRowPtrB, cl_mem csrColIndB, cl_mem csrRowCReorder, cl_mem csrRowCNnzSize, std::vector<int> intBin, std::vector<int> intPtr, int max_intprod, clsparseControl control)
{
    if(intBin[0] > 0)
        compute_nnzC_0(csrRowCReorder, csrRowCNnzSize, intBin[0], control);
    if(intBin[1] > 0)
        compute_nnzC_1(csrRowCReorder, csrRowCNnzSize, intBin[1], intPtr[1], control);
    int i;
    for(i = 2; i <= 7; i++)
    {
        if(intBin[i] > 0)
            compute_nnzC_pwarp(csrRowPtrA, csrColIndA, csrRowPtrB, csrColIndB, csrRowCReorder, csrRowCNnzSize, intBin[i], intPtr[i], 1 << (i - 1), control);
    }
    for(i = 8; i < 12; i++)
    {
        if(intBin[i] > 0)
            compute_nnzC_tb(64, csrRowPtrA, csrColIndA, csrRowPtrB, csrColIndB, csrRowCReorder, csrRowPtrCSize, intBin[i], intPtr[i], control);
    }
        
    for(i = 12; i < NUM_SEGMENTS - 1; i++)
    {
        if(intBin[i] > 0)
        {
            int num_threads = 1 << (i - 6);
            compute_nnzC_tb(num_threads, csrRowPtrA, csrColIndA, csrRowPtrB, csrColIndB, csrRowCReorder, csrRowPtrCSize, intBin[i], intPtr[i], control);
        }
    }
    if(intBin[15] > 0)
    {
        int pattern = 0;

        cl_mem clfail_count = ::clCreateBuffer( cxt(), CL_MEM_READ_WRITE, sizeof(cl_int), NULL, &run_status );

        clEnqueueFillBuffer(control->queue(), clfail_count, &pattern, sizeof(cl_int), 0, sizeof(cl_int), 0, NULL, NULL);

        cl_mem clfail_perm = ::clCreateBuffer( cxt(), CL_MEM_READ_WRITE, intBin[15] * sizeof(cl_int), NULL, &run_status );

        clEnqueueFillBuffer(control->queue(), clfail_perm, &pattern, sizeof(cl_int), 0, intBin[15] * sizeof(cl_int), 0, NULL, NULL);

        compute_nnzC_tb_large(csrRowPtrA, csrColIndA, csrRowPtrB, csrColIndB, csrRowCReorder, csrRowPtrCNnzSize, clfail_count, clfail_perm, intBin[15], ptr, control);

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

        if(fail_count > 0)
        {
            int minuspattern = -1;
            cl_mem clhashtable = ::clCreateBuffer( cxt(), CL_MEM_READ_WRITE, fail_max * fail_count * sizeof(cl_int), NULL, &run_status );
            clEnqueueFillBuffer(control->queue(), clfail_perm, &minuspattern, sizeof(cl_int), 0, max_intprod * fail_count * sizeof(cl_int), 0, NULL, NULL);
            compute_nnzC_global(csrRowPtrA, csrColIndA, csrRowPtrB, csrColIndB, clfail_perm, csrRowrCNnzSize, clhashtable, fail_count, control);
            ::clReleaseMemObject(clhashtable);
        }

        ::clReleaseMemObject(clfail_count);
        ::clReleaseMemObject(clfail_max);
        ::clReleaseMemObject(clfail_perm);
    }

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

clsparseStatus compute_nnzC_0(cl_mem csrRowCReorder, cl_mem csrRowCNnzSize, int bin, clsparseControl control)
{
    const std::string params = std::string() +
               "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type
            + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type;


    cl::Kernel kernel = KernelCache::get(control->queue,"SpGEMM_computeNnz_kernels", "compute_nnzC_0", params);

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    int num_threads = GROUPSIZE_256;
    size_t num_blocks = ceil((double)bin / (double)num_threads);

    szLocalWorkSize[0]  = GROUPSIZE_256;
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

clsparseStatus compute_nnzC_1(cl_mem csrRowCReorder, cl_mem csrRowCNnzSize, int bin, int ptr, clsparseControl control)
{
    const std::string params = std::string() +
               "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type
            + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type;


    cl::Kernel kernel = KernelCache::get(control->queue,"SpGEMM_computeNnz_kernels", "compute_nnzC_1", params);

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    int num_threads = GROUPSIZE_256;
    size_t num_blocks = ceil((double)bin / (double)num_threads);

    szLocalWorkSize[0]  = GROUPSIZE_256;
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

clsparseStatus compute_nnzC_pwarp(cl_mem csrRowPtrA, cl_mem csrColIndA, cl_mem csrRowPtrB, cl_mem csrColIndB, cl_mem csrRowCReorder, cl_mem csrRowCNnzSize, int bin, int ptr, int intSize, clsparseControl control)
{
    const std::string params = std::string() +
               "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type
            + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type;


    cl::Kernel kernel = KernelCache::get(control->queue,"SpGEMM_computeNnz_kernels", "compute_nnzC_pwarp", params);

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    int num_threads = 64;
    size_t num_blocks = ceil((double)bin / (double)num_threads);

    szLocalWorkSize[0]  = 64;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    KernelWrap kWrapper(kernel);
    kWrapper << csRowPtrA << csColIndA << csrRowPtrB <<  csrColIndB <<  csrRowCReorder <<  csrRowCNnzSize << cl::Local(64 * sizeof(int) ) << bin << ptr << intSize;

    cl::NDRange local(szLocalWorkSize[0]);
    cl::NDRange global(szGlobalWorkSize[0]);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

clsparseStatus compute_nnzC_tb(int num_threads, cl_mem csrRowPtrA, cl_mem csrColIndA, cl_mem csrRowPtrB, cl_mem csrColIndB, cl_mem csrRowCReorder, cl_mem csrRowCNnzSize, int bin, int ptr, int intSize, clsparseControl control)
{
    const std::string params = std::string() +
               "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type
            + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type;


    cl::Kernel kernel = KernelCache::get(control->queue,"SpGEMM_computeNnz_kernels", "compute_nnzC_tb", params);

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    size_t num_blocks = ceil((double)bin / (double)num_threads);

    szLocalWorkSize[0]  = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    KernelWrap kWrapper(kernel);
    kWrapper << csRowPtrA << csColIndA << csrRowPtrB <<  csrColIndB <<  csrRowCReorder <<  csrRowCNnzSize << cl::Local(intSize * sizeof(int) ) << bin << ptr << intSize;

    cl::NDRange local(szLocalWorkSize[0]);
    cl::NDRange global(szGlobalWorkSize[0]);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

clsparseStatus compute_nnzC_tb_large(cl_mem csrRowPtrA, cl_mem csrColIndA, cl_mem csrRowPtrB, cl_mem csrColIndB, cl_mem csrRowCReorder, cl_mem csrRowCNnzSize, cl_mem fail_count, cl_mem fail_perm, int bin, int ptr, clsparseControl control)
{
    const std::string params = std::string() +
               "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type
            + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type;


    cl::Kernel kernel = KernelCache::get(control->queue,"SpGEMM_computeNnz_kernels", "compute_nnzC_tb_large", params);

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    int num_threads = GROUPSIZE_256;
    size_t num_blocks = ceil((double)bin / (double)num_threads);

    szLocalWorkSize[0]  = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    KernelWrap kWrapper(kernel);
    kWrapper << csRowPtrA << csColIndA << csrRowPtrB <<  csrColIndB <<  csrRowCReorder << csrRowCNnzSize << fail_count << fail_perm << cl::Local((MAX_HASH_SIZE - 1) * sizeof(int)) << cl::Local(sizeof(int)) << bin << ptr;

    cl::NDRange local(szLocalWorkSize[0]);
    cl::NDRange global(szGlobalWorkSize[0]);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

clsparseStatus compute_nnzC_tb_global(cl_mem csrRowPtrA, cl_mem csrColIndA, cl_mem csrRowPtrB, cl_mem csrColIndB, cl_mem fail_perm, cl_mem csrRowCNnzSize, cl_mem clhashtable, int fail_count, int max_intprod, control)
{
    const std::string params = std::string() +
               "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type
            + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type;


    cl::Kernel kernel = KernelCache::get(control->queue,"SpGEMM_computeNnz_kernels", "compute_nnzC_tb_large", params);

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    int num_threads = GROUPSIZE_256;
    size_t num_blocks = ceil((double)bin / (double)num_threads);

    szLocalWorkSize[0]  = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    KernelWrap kWrapper(kernel);
    kWrapper << csRowPtrA << csColIndA << csrRowPtrB <<  csrColIndB << fail_perm << csrRowCNnzSize << clhashtable << fail_count << max_intprod;

    cl::NDRange local(szLocalWorkSize[0]);
    cl::NDRange global(szGlobalWorkSize[0]);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}


 CLSPARSE_EXPORT clsparseStatus
        clsparseScsrSpGemm(
        const clsparseCsrMatrix* sparseMatA,
        const clsparseCsrMatrix* sparseMatB,
              clsparseCsrMatrix* sparseMatC,
        const clsparseControl control )
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

    const clsparseCsrMatrixPrivate* matA = static_cast<const clsparseCsrMatrixPrivate*>(sparseMatA);
    const clsparseCsrMatrixPrivate* matB = static_cast<const clsparseCsrMatrixPrivate*>(sparseMatB);
    clsparseCsrMatrixPrivate* matC = static_cast<clsparseCsrMatrixPrivate*>(sparseMatC);

    size_t m = matA->num_rows;
    size_t k1 = matA->num_cols;
    size_t k2 = matB->num_rows;
    size_t n  = matB->num_cols;
    size_t nnzA = matA->num_nonzeros;
    size_t nnzB = matB->num_nonzeros;

    if(k1 != k2)
    {
        std::cerr << "A.n and B.m don't match!" << std::endl;
        return clsparseInvalidKernelExecution;
    }

    cl_mem csrRowPtrA = matA->row_pointer;
    cl_mem csrColIndA = matA->col_indices;
    cl_mem csrValA    = matA->values;
    cl_mem csrRowPtrB = matB->row_pointer;
    cl_mem csrColIndB = matB->col_indices;
    cl_mem csrValB    = matB->values;

    cl::Context cxt = control->getContext();

    // STAGE 1 Count the number of intermediate products of each row
    int pattern = 0;

    // interproduct numbers of each C's row
    cl_mem csrRowCIntProdNum = ::clCreateBuffer( cxt(), CL_MEM_READ_WRITE, m * sizeof( cl_int ), NULL, &run_status );
    clEnqueueFillBuffer(control->queue(), csrRowPtrCIntSize, &pattern, sizeof(cl_int), 0, m * sizeof(cl_int), 0, NULL, NULL);
    // max interproduct number of all rows
    cl_mem clMaxIntProd = ::clCreateBuffer( cxt(), CL_MEM_READ_WRITE, sizeof( cl_int ), NULL, &run_status );
    clEnqueueFillBuffer(control->queue(), csrRowPtrCIntSize, clMaxIntProd, sizeof(cl_int), 0, *sizeof(cl_int), 0, NULL, NULL);

    compute_nnzCIntProdNum(m, csrRowPtrA, csrColIndA, csrRowPtrB, csrColIndB, csrRowCIntProdNum, clMaxIntProd, control);

    int max_intprod;

    run_status = clEnqueueReadBuffer(control->queue(),
                                     clMaxIntProd,
                                     1,
                                     0,
                                     sizeof(int),
                                     &max_intprod,
                                     0,
                                     0,
                                     0);
    
    ::clReleaseMemObject(clMaxIntProd);

    // STAGE 2 Divide the rows into groups by the number of intermediate products
    // bins of rows with interproduct numbers
    // local memory size: 32KB.
    // FP16: index 4byte + value 2byte => Max 8192 indexes or 5461 index and value pair
    // FP32: index 4byte + value 4byte => Max 8192 indexes or 4096 index and value pair
    // FP64: index 4byte + value 8byte => Max 8192 indexes or 2730 index and value pair
    // bin: {0, 1, 2, 3~4, 5~8, 9~16, 17~32, 33~64, 65~128, 129~256, 257~512, 513~1024, 1025~2048, 2049~4096, 4097~8192, 8192~}
    cl_mem clIntBin = ::clCreateBuffer( cxt(), CL_MEM_READ_WRITE, INT_PROD_NUM_SEGMENTS * sizeof(cl_int), NULL, &run_status );
    clEnqueueFillBuffer(control->queue(), clIntBin, &pattern, sizeof(cl_int), 0, INT_PROD_NUM_SEGMENTS * sizeof(cl_int), 0, NULL, NULL);

    compute_nnzIntBin(m, csrRowCIntProdNum, clIntBin, control);

    std::vector<int> intBin(INT_PROD_NUM_SEGMENTS, 0);

    run_status = clEnqueueReadBuffer(control->queue(),
                                     clintBin,
                                     1,
                                     0,
                                     INT_PROD_NUM_SEGMENTS * sizeof(int),
                                     intBin.data(),
                                     0,
                                     0,
                                     0);
    ::clReleaseMemObject(clIntBin);

    char *intprodstring[INT_PROD_NUM_SEGMENTS] = {"0", "1", "2", "3~4", "5~8", "9~16", "17~32", "33~64", "65~128", "129~256", "257~512", "513~1024", "1025~2048", "2049~4096", "4097~8192", "8193~"};

    for(int i = 0; i < INT_PROD_NUM_SEGMENTS; i++)
        printf("%s: %d\n", intprodstring[i], intBin[i]);

    // STAGE 3 Reorder rows of C based on the bins
    std::vector<int> intPtr(INT_PROD_NUM_SEGMENTS, 0);

    int i;
    for(i = 1; i < INT_PROD_NUM_SEGMENTS; i++)
        intPtr[i] = intPtr[i - 1] + intBin[i - 1];

    cl_mem clIntPtr = ::clCreateBuffer( cxt(), CL_MEM_READ_WRITE, INT_PROD_NUM_SEGMENTS * sizeof(cl_int), NULL, &run_status );
    run_status = clEnqueueWriteBuffer(control->queue(),
                                     clintPtr,
                                     1,
                                     0,
                                     INT_PROD_NUM_SEGMENTS * sizeof(int),
                                     intPtr.data(),
                                     0,
                                     0,
                                     0);
    cl_mem csrRowCReorder = ::clCreateBuffer( cxt(), CL_MEM_READ_WRITE, m * sizeof( cl_int ), NULL, &run_status );
    clEnqueueFillBuffer(control->queue(), csrRowCReorder, &pattern, sizeof(cl_int), 0, m * sizeof(cl_int), 0, NULL, NULL);

    compute_reorderRow(m, csrRowCIntProdNum, clIntPtr, csrRowCReorder, control);

    ::clReleaseMemObject(clIntPtr);

    // STAGE 4  Count the number of non-zero elements of each row of output matrix for all groups
    cl_mem csrRowCNnzSize = ::clCreateBuffer( cxt(), CL_MEM_READ_WRITE, m * sizeof( cl_int ), NULL, &run_status );
    clEnqueueFillBuffer(control->queue(), csrRowCNnzSize, &pattern, sizeof(cl_int), 0, (m + 1)*sizeof(cl_int), 0, NULL, NULL);

    compute_nnzC(csrRowPtrA, csrColIndA, csrRowPtrB, csrColIndB, csrRowCReorder, csrRowCIntProdNum, csrRowCNnzSize, intBin, intPtr, max_intprod, control);

    ::clReleaseMemObject(csrRowCIntProdNum);
    
    matC->row_pointer = ::clCreateBuffer( cxt(), CL_MEM_READ_WRITE, (m + 1) * sizeof( cl_int ), NULL, &run_status );
    clEnqueueFillBuffer(control->queue(), matC->row_pointer, &pattern, sizeof(cl_int), 0, (m + 1)*sizeof(cl_int), 0, NULL, NULL);
    cl_mem csrRowPtrC = matC->row_pointer;
    matC->col_indices = ::clCreateBuffer( cxt(), CL_MEM_READ_WRITE, nnzC * sizeof( cl_int ), NULL, &run_status );
    matC->values =     ::clCreateBuffer( cxt(), CL_MEM_READ_WRITE, nnzC * sizeof( cl_float ), NULL, &run_status );
   
    matC->num_rows = m;
    matC->num_cols = n;
    matC->num_nonzeros  = nnzC;

}

