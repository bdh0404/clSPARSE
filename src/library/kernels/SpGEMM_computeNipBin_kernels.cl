
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

//#define GROUPSIZE_256 256

#define NIP_SEGMENTS 16

/**
 * @brief Count the number of rows in each intermediate products' bin
 * 
 * @param d_csrRowCInnProdNum Number of products in each C's row
 * @param s_innBin Local memory buffer of bins of C's rows
 * @param d_innBin Bins of C's rows
 * @param m Number of C's rows
 */
__kernel
void compute_nipBin_kernel(
        __global const int *d_csrRowCInnProdNum,
        __local int *s_innBin,
        __global int *d_innBin,
        const int m)
{
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int i;

    // initialize the local buffer to 0
    if(local_id < NIP_SEGMENTS)
        s_innBin[local_id] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    if(global_id < m)
    {
        int innSize =  d_csrRowCInnProdNum[global_id];
        // add the count in each bin with atomic_add on local memory
        if(innSize == 0)
            atomic_add(&s_innBin[0], 1);
        else if(innSize == 1)
            atomic_add(&s_innBin[1], 1);
        else if(innSize == 2)
            atomic_add(&s_innBin[2], 1);
        else if(innSize > 8192)
            atomic_add(&s_innBin[NIP_SEGMENTS - 1], 1);
        else
        {
            for(i = 3; i < NIP_SEGMENTS - 1; i++)
            {
                if((1 << (i - 2)) < innSize && innSize <= (1 << (i - 1)))
                {
                    atomic_add(&s_innBin[i], 1);
                    break;
                }
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    // add the result into global memory
    if(local_id < NUM_SEGMENTS)
    {
        int innBinPartial = s_innBin[local_id];
        atomic_add(&d_innBin[local_id], innBinPartial);
    }
}