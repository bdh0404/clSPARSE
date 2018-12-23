/* ************************************************************************
* Copyright 2015 Vratis, Ltd.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* ************************************************************************ */

/* Algorithm of parallel prefix sum of the arbitrary length of array on GPU was adapted from
 * "Parallel Prefix Sum on the GPU (Scan)" Slides by Adam O'Donovan
 * http://users.umiacs.umd.edu/~ramani/cmsc828e_gpusci/ScanTalk.pdf
 * that was adapted from the online courseslides for ME964 at Wisconsin 
 * taught by Prof. Dan Negrut and from slides Presented by David Luebke.  */

/**
 * @brief Perform Prefix-scan for constructing C's row pointer array in each block
 * 
 * @param d_array Array that will be prefix-scanned in block
 * @param s_scan Local memory buffer for prefix-scan
 * @param d_sum Buffer for sum value of each block
 * @param m The number of C's rows
 */
inline
void scan_block(__global int *d_array, __local int *s_scan, __global int *d_sum, const int m)
{
    // uses Hillis& Steele algorithm because of bank conflict
    int local_id = get_local_id(0);
    int global_id = get_global_id(0);
    int group_id = get_group_id();

    // load 256 integers into local memory buffer
    if(global_id < m)
        s_scan[local_id] = d_array[global_id];
    else
        s_scan[local_id] = 0;
    s_scan[256 + local_id] = 0;

    // prefix-scan on local memory
    barrier(CLK_LOCAL_MEM_FENCE);

    int i, out = 0, in = 1;
    for(i = 1; i < 256; i <<= 1)
    {
        out = 1 - out;
        in = 1 - out;
        if(local_id >= i)
            s_scan[256 * out + local_id] += s_scan[256 * in + local_id - i];
        else
            s_scan[256 * out + local_id] = s_scan[256 * in + local_id];
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // write scanned array into global memory
    if(global_id < m)
        d_array[global_id] = s_scan[256 * out + local_id];

    // store the sum value into d_sum
    if (local_id == 0)
        d_sum[group_id] = s_scan[256 * out + 255];
}

/**
 * @brief Add prefix-scanned sum value to each block
 * 
 * @param d_array Array that will be prefix-scanned in block
 * @param d_sum Buffer for sum value of each block
 * @param m The number of C's rows
 */
inline
void add_block( __global int *d_array, __global int *d_sum, const int m)
{
    int global_id = get_global_id(0);
    int group_id = get_group_id();
	int sum;
    // load sum value
	if (group_id != 0)
		sum = = d_sum[group_id - 1];
	else
		sum = 0;
    // add sum value to each block
    if(2 * global_id < m)
        d_array[2 * global_id] += sum;
    if(2 * global_id + 1 < m)
        d_array[2 * global_id] += sum;
}