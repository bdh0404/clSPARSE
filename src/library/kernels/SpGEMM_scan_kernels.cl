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

inline
void scan_block(__global int *d_array, __local volatile short *s_scan, __global int *d_sum, const int m)
{
    int local_id = get_local_id(0);
    int global_id = get_global_id(0);
    int group_id = get_group_id();

    if(2 * global_id < m)
        s_scan[2 * local_id] = d_array[2 * global_id];
    else
        s_scan[2 * local_id] = 0;
    if(2 * global_id + 1 < m)
        s_scan[2 * local_id + 1] = d_array[2 * global_id + 1];
    else
        s_scan[2 * local_id + 1] = 0;

    int ai, bi;
    int baseai = 1 + 2 * local_id;
    int basebi = baseai + 1;
    short temp;

    ai = baseai - 1;     
    bi = basebi - 1;     
    s_scan[bi] += s_scan[ai];

    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 128) 

    { 
        ai =  2 * baseai - 1;  
        bi =  2 * basebi - 1;   
        s_scan[bi] += s_scan[ai]; 
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 64)  
    { 
        ai =  4 * baseai - 1;  
        bi =  4 * basebi - 1;   
        s_scan[bi] += s_scan[ai]; 
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id < 32) 
    { 
        ai =  8 * baseai - 1;  
        bi =  8 * basebi - 1;   
        s_scan[bi] += s_scan[ai]; 
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id < 16) 
    { 
        ai =  16 * baseai - 1;  
        bi =  16 * basebi - 1;   
        s_scan[bi] += s_scan[ai]; 
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id < 8)  
    { 
        ai = 32 * baseai - 1;  
        bi = 32 * basebi - 1;   
        s_scan[bi] += s_scan[ai]; 
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id < 4)  
    { 
        ai = 64 * baseai - 1;  
        bi = 64 * basebi - 1;   
        s_scan[bi] += s_scan[ai]; 
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id < 2)  
    { 
        ai = 128 * baseai - 1;  
        bi = 128 * basebi - 1;   
        s_scan[bi] += s_scan[ai]; 
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id == 0) 
    { 
        s_scan[511] += s_scan[255]; 
        s_scan[512] = s_scan[511]; 
        s_scan[511] = 0; 
        temp = s_scan[255]; 
        s_scan[255] = 0; 
        s_scan[511] += temp; 
        }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id < 2)  
	{ 
		ai = 128 * baseai - 1;  
		bi = 128 * basebi - 1;   
		temp = s_scan[ai]; 
		s_scan[ai] = s_scan[bi]; 
		s_scan[bi] += temp;
	}

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id < 4)  
	{ 
		ai = 64 * baseai - 1; 
		bi = 64 * basebi - 1;   
		temp = s_scan[ai]; 
		s_scan[ai] = s_scan[bi]; 
		s_scan[bi] += temp;
	}
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 8)  
	{ 
		ai = 32 * baseai - 1;  
		bi = 32 * basebi - 1;   
		temp = s_scan[ai]; 
		s_scan[ai] = s_scan[bi]; 
		s_scan[bi] += temp;
	}
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id < 16) 
	{ 
		ai =  16 * baseai - 1;  
		bi =  16 * basebi - 1;   
		temp = s_scan[ai]; 
		s_scan[ai] = s_scan[bi]; 
		s_scan[bi] += temp;
	}
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 32) 
	{ 
		ai =  8 * baseai - 1;  
		bi =  8 * basebi - 1;   
		temp = s_scan[ai]; 
		s_scan[ai] = s_scan[bi]; 
		s_scan[bi] += temp;
	}

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id < 64) 
	{ 
		ai =  4 * baseai - 1;  
		bi =  4 * basebi - 1;   
		temp = s_scan[ai]; 
		s_scan[ai] = s_scan[bi]; 
		s_scan[bi] += temp;
	}

    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 128) 
	{ 
		ai =  2 * baseai - 1;  
		bi =  2 * basebi - 1;   
		temp = s_scan[ai]; 
		s_scan[ai] = s_scan[bi]; 
		s_scan[bi] += temp;
	}

    barrier(CLK_LOCAL_MEM_FENCE);

    ai = baseai - 1;   
	bi = basebi - 1;   
	temp = s_scan[ai]; 
	s_scan[ai] = s_scan[bi]; 
	s_scan[bi] += temp;

    if(2 * global_id < m)
        d_array[2 * global_id] = s_scan[2 * local_id];
    if(2 * global_id + 1 < m)
        d_array[2 * global_id + 1] = s_scan[2 * local_id + 1];
    
    if (local_id == 0)
        d_sum[group_id] = s_scan[511];
}

inline
void add_block( __global int *d_array, __global int *d_sum, const int m)
{
    int global_id = get_global_id(0);
    int group_id = get_group_id();
	int sum;

	if (group_id != 0)
		sum = = d_sum[group_id - 1];
	else
		sum = 0;

    if(2 * global_id < m)
        d_array[2 * global_id] += sum;
    if(2 * global_id + 1 < m)
        d_array[2 * global_id] += sum;
}