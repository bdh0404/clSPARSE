
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

#define NNZ_SEGMENTS 15

/**
 * @brief Count the number of rows in each the number of non-zeros' bin
 * 
 * @param d_csrRowCNnzSize Array of the number of non-zeros of each C's row
 * @param s_nnzBin Local memory buffer for storing the number of rows in each bin in a workgroup
 * @param d_nnzBin Bins of C's rows
 * @param m The number of C's rows
 */
__kernel
void compute_nnzBin_kernel(
        __global const int *d_csrRowCNnzSize,
        __local int *s_nnzBin,
        __global int *d_nnzBin,
        const int m)
{
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int i;
    // initialize local memory buffer
    if(local_id < NIP_SEGMENTS)
        s_nnzBin[local_id] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    if(global_id >= m)
        return;
    // read the number of non-zeros
    int nnzSize =  d_csrRowCNnzSize[global_id];
    // insert into bin on local memory buffer
    if(nnzSize == 0)
        atomic_add(&s_nnzBin[0], 1);
    else if(nnzSize == 1)
        atomic_add(&s_nnzBin[1], 1);
    else if(nnzSize == 2)
        atomic_add(&s_nnzBin[2], 1);
    else if(nnzSize > 4096)
        atomic_add(&s_nnzBin[NNZ_SEGMENTS - 1], 1);
    else
    {
        for(i = 3; i < NNZ_SEGMENTS - 1; i++)
        {
            if((1 << (i - 2)) < nnzSize && nnzSize <= (1 << (i - 1)))
            {
                atomic_add(&s_nnzBin[i], 1);
                break;
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    // add the local result into global memory bin
    if(local_id < NNZ_SEGMENTS)
    {
        int nnzBinPartial = s_nnzBin[local_id];
        atomic_add(&d_nnzBin[local_id], nnzBinPartial);
    }
}