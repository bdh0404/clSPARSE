
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

/**
 * @brief Compute the number of products of each C's row
 * 
 * @param d_csrRowPtrA A's row pointer array
 * @param d_csrColIndA A's column indices array
 * @param d_csrRowPtrB B's row pointer array
 * @param d_csrRowCInnProdNum Buffer for storing the result
 * @param s_max_intprod Local memory buffer for storing the maximum number of products in a workgroup
 * @param d_max_inn_prod Buffer for storing the maximum number of products of all C's rows
 * @param m Number of C's rows
 */
__kernel
void compute_InnProdNum_kernel(
        __global const int *d_csrRowPtrA,
        __global const int *d_csrColIndA,
        __global const int *d_csrRowPtrB,
        __global int *d_csrRowCInnProdNum,
        __local int *s_max_intprod,
        __global int *d_max_inn_prod,
        const int m)
{
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int start, stop, index, strideB, row_size_Ct = 0;

    if (global_id < m)
    {
        start = d_csrRowPtrA[global_id];
        stop = d_csrRowPtrA[global_id + 1];
        // for each column index in a row
        for (int i = start; i < stop; i++)
        {
            index = d_csrColIndA[i];
            // add the corresponding size of B's row size
            strideB = d_csrRowPtrB[index + 1] - d_csrRowPtrB[index];
            row_size_Ct += strideB;
        }
        // write the result
        d_csrRowCInnProdNum[global_id] = row_size_Ct;
    }

    // calculate maximum number in each local memory first,
    s_max_intprod[local_id] = row_size_Ct;

    barrier(CLK_LOCAL_MEM_FENCE);

    int i = 128;
    // compare numbers in local memory and the final result will be stored in s_max_intprod[0]
    while(i > 0)
    {
        if(local_id < i)
            s_max_intprod[local_id] = max(s_max_intprod[local_id], s_max_intprod[local_id + i]);
        
        i >>= 1;
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(local_id == 0)
    {
        // and calculate real maximum number in GLOBAL memory
        atomic_max(d_max_inn_prod, s_max_intprod[0]);
    }
}