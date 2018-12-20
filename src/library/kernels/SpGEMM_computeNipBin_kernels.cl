
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
#if defined(cl_khr_global_int32_base_atomics) && defined(cl_khr_global_int32_extended_atomics)
    #pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
    #pragma OPENCL_EXTENSION cl_khr_global_int32_extended_atomics : enable
#else
    #error "Required 32-bit atomics not supported by this OpenCL implemenation."
#endif

#define NIP_SEGMENTS 16

__kernel
void compute_nipBin_kernel(
        __global const int *d_csrRowCIntProdNum,
        __local int *s_intBin,
        __global int *d_intBin,
        const int m)
{
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int i;

    if(local_id < NIP_SEGMENTS)
        s_intBin[local_id] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    if(global_id >= m)
        return;

    int intSize =  d_csrRowCIntProdNum[global_id];

    if(intSize == 0)
        atomic_add(&s_intBin[0], 1);
    else if(intSize == 1)
        atomic_add(&s_intBin[1], 1);
    else if(intSize == 2)
        atomic_add(&s_intBin[2], 1);
    else if(intSize > 8192)
        atomic_add(&s_intBin[NIP_SEGMENTS - 1], 1);
    else
    {
        for(i = 3; i < NIP_SEGMENTS - 1; i++)
        {
            if((1 << (i - 2)) < intSize && intSize <= (1 << (i - 1)))
            {
                atomic_add(&s_intBin[i], 1);
                break;
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if(local_id < NUM_SEGMENTS)
    {
        int intBinPartial = s_intBin[local_id];
        atomic_add(&d_intBin[local_id], intBinPartial);
    }
}

