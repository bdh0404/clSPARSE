#define HASH_CONST 127
#define MAX_HASH_SIZE 8192

/* Atomic float add operations from
 * http://simpleopencl.blogspot.com/2013/05/atomic-operations-and-floats-in-opencl.html */
void atomic_add_global(volatile global float *source, const float operand) 
{
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;
 
    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    } while (atomic_cmpxchg((volatile global unsigned int *)source, prevVal.intVal, newVal.intVal) 
		!= prevVal.intVal);
}

void atomic_add_local(volatile local float *source, const float operand) 
{
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;
 
    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;
 
    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    } while (atomic_cmpxchg((volatile local unsigned int *)source, prevVal.intVal, newVal.intVal) 
		!= prevVal.intVal);
}

/**
 * @brief Compute columns and values of C's rows with 1 non-zero
 * 
 * @param d_csrRowPtrA A's row pointer array
 * @param d_csrColIndA A's column indices array
 * @param d_csrValA  A's value array
 * @param d_csrRowPtrB B's row pointer array
 * @param d_csrColIndB B's column indices array
 * @param d_csrValB B's value array
 * @param d_csrRowPtrC C's row pointer array
 * @param d_csrColIndC C's column indices array
 * @param d_csrValC C's value array
 * @param d_csrRowCReorder Reordered C's rows
 * @param bin Number of C's rows with 1 non-zero
 * @param ptr Baseline pointer for the non-zero bin
 */
__kernel
void compute_valC_1(
        __global const int *d_csrRowPtrA,
        __global const int *d_csrColIndA,
        __global const float *d_csrValA,
        __global const int *d_csrRowPtrB,
        __global const int *d_csrColIndB,
        __global const float *d_csrValB,
        __global int *d_csrRowPtrC,
        __global int *d_csrColIndC,
        __global float *d_csrValC,
        __global const int *d_csrRowCReorder,
        const int bin,
        const int ptr)
{
    int global_id = get_global_id(0);

    if(global_id >= bin) 
		return;
	// find row id
    int row_id = d_csrRowCReorder[ptr + global_id];
	// find the location
    int offset = d_csrRowPtrC[row_id];
	// find each values
    int acol = d_csrColIndA[row_id];
    float aval = d_csrValA[row_id];
    int brow = d_csrRowPtrB[acol];
    int bcol = d_csrColIndB[brow];
    float bval = d_csrValB[brow];
	// set the values
    d_csrColIndC[offset] = bcol;
    d_csrValC[offset] = aval * bval;
}

/**
 * @brief Compute columns and values of C's rows with 2~32 non-zeros
 * 
 * @param d_csrRowPtrA A's row pointer array
 * @param d_csrColIndA A's column indices array
 * @param d_csrValA  A's value array
 * @param d_csrRowPtrB B's row pointer array
 * @param d_csrColIndB B's column indices array
 * @param d_csrValB B's value array
 * @param d_csrRowPtrC C's row pointer array
 * @param d_csrColIndC C's column indices array
 * @param d_csrValC C's value array
 * @param d_csrRowCReorder Reordered C's rows
 * @param s_Nz Local memory buffer for storing the number of non-zeros
 * @param s_colhashtable Local memory hash table for column index matching
 * @param s_valhashtable Local memory hash table for value addition
 * @param bin Number of C's rows with 1 non-zero
 * @param ptr Baseline pointer for the non-zero bin
 * @param nnzSize maximum non-zeros in each bin
 */
__kernel
void compute_valC_pwarp(
        __global const int *d_csrRowPtrA,
        __global const int *d_csrColIndA,
        __global const float *d_csrValA,
        __global const int *d_csrRowPtrB,
        __global const int *d_csrColIndB,
        __global const float *d_csrValB,
        __global int *d_csrRowPtrC,
        __global int *d_csrColIndC,
        __global float *d_csrValC,
        __global const int *d_csrRowCReorder,
        __local int *s_Nz,
        __local int *s_colhashtable,
		__local float *s_valhashtable,
        const int bin,
        const int ptr,
        const int nnzSize)
{
	int global_id = get_global_id(0);
	int rid = global_id / nnzSize;
	int local_id = get_local_id(0);
	int num_rows = 64 / nnzSize;
	int nz_id = rid % num_rows;

	// initialize local memory buffers
	s_colhashtable[local_id] = -1;
	s_colhashtable[local_id + 64] = -1;
	s_valhashtable[local_id] = 0;
	s_valhashtable[local_id + 64] = 0;

	if(local_id < num_rows)
		s_Nz[local_id] = 0;

	barrier(CLK_LOCAL_MEM_FENCE);

	if (rid >= bin)
		return;

	int start_col_index_A, stop_col_index_A;
	int start_col_index_B, stop_col_index_B;
	int tid = global_id % nnzSize;
	int hash_size = 2 * nnzSize;
	int local_rid = nz_id * hash_size;
	// find the row
	rid = d_csrRowCReorder[ptr + rid];

	start_col_index_A = d_csrRowPtrA[rid];
	stop_col_index_A = d_csrRowPtrA[rid + 1];

	int i, nnz = 0;

	// for each column index of A
	for (i = d_csrRowPtrA[rid] + tid; i < d_csrRowPtrA[rid + 1]; i += nnz_size)
	{
		// find the corresponding row of B
		int row_id_b = d_csrColIndA[i];
		int aval = d_csrValA[i];

		start_col_index_B = d_csrRowPtrB[row_id_B];
		stop_col_index_B = d_csrRowPtrB[row_id_B + 1];

		int j;
		// for each column index of B
		for (j = start_col_index_B; j < stop_col_index_B; j++)
		{
			// perform hash operation
			int key = d_csrColIndB[j];
			int bval = d_csrValB[j];
			int hash = (bcol * HASH_CONST) % hash_size;
			int addr = local_rid + hash;
			// add to hash table
			while (true)
			{
				if (s_colhashtable[addr] == key)
				{
					atomic_add_local(&s_valhashtable[addr], aval * bval);
					break;
				}
				else if (scolhashtable[addr] == -1)
				{
					int old = atomic_cmpxchg(&s_colhashtable[addr], -1, key);

					if (old == -1)
					{
						atomic_add_local(&s_valhashtable[addr], aval * bval);

						break;
					}
					else
					{
						hash = (hash + 1) % hash_size;
						addr = local_rid + hash;
					}
				}
			}
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	// Compact the results into the front of each hash table and count the number of non-zeros
	for (i = tid; i < hash_size; i += nnzSize)
	{
		int index, ccol, cval;

		if (s_colhashtable[local_rid + i] != -1)
		{
			index = atomic_add(s_Nz + nz_id, 1);
			ccol = s_colhashtable[local_rid + i];
			cval = s_valhashtable[local_rid + i];
		}
		else
			index = -1;

		barrier(CLK_LOCAL_MEM_FENCE);

		if (index != -1)
		{
			s_colhashtable[local_rid + index] = ccol;
			s_valhashtable[local_rid + index] = cval;
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	// sort and store the results into global memory
	int nnz = s_Nz[nz_id];

	if (tid < nnz)
	{
		int target = s_colhashtable[local_rid + tid];
		int count = 0;
		// find the exact location
		for (j = 0; j < nnz; j++)
			count += (unsigned int)(s_colhashtable[local_rid + j] - target) >> 31;

		int offset = d_csrRowPtrC[rid];
		d_csrColIndC[offset + count] = s_colhashtable[local_rid + tid];
		d_csrValC[offset + count] = s_valhashtable[local_rid + tid];
	}
}

/**
 * @brief Compute columns and values of C's rows with 33~2048 non-zeros
 * 
 * @param d_csrRowPtrA A's row pointer array
 * @param d_csrColIndA A's column indices array
 * @param d_csrValA  A's value array
 * @param d_csrRowPtrB B's row pointer array
 * @param d_csrColIndB B's column indices array
 * @param d_csrValB B's value array
 * @param d_csrRowPtrC C's row pointer array
 * @param d_csrColIndC C's column indices array
 * @param d_csrValC C's value array
 * @param d_csrRowCReorder Reordered C's rows
 * @param s_Nz Local memory buffer for storing the number of non-zeros
 * @param s_colhashtable Local memory hash table for column index matching
 * @param s_valhashtable Local memory hash table for value addition
 * @param bin Number of C's rows with 1 non-zero
 * @param ptr Baseline pointer for the non-zero bin
 * @param nnzSize maximum non-zeros in each bin
 */
__kernel
void compute_valC_tb(
        __global const int *d_csrRowPtrA,
        __global const int *d_csrColIndA,
        __global const float *d_csrValA,
        __global const int *d_csrRowPtrB,
        __global const int *d_csrColIndB,
        __global const float *d_csrValB,
        __global int *d_csrRowPtrC,
        __global int *d_csrColIndC,
        __global float *d_csrValC,
        __global const int *d_csrRowCReorder,
        __local int *s_Nz,
        __local int *s_colhashtable,
		__local float *s_valhashtable,
        const int bin,
        const int ptr,
        const int nnzSize)
{
	int rid = get_group_id(0);
	int tid = get_local_id(0);
	int local_size = get_local_size(0);
	int hash_size = 2 * nnzSize - 1;

	int i;
	
	// initialize local memory hash tables
	for(i = tid; i < hash_size; i += local_size)
	{
		s_colhashtable[i] = -1;
		s_valhashtable[i] = 0;
	}
	if(local_id == 0)
		s_Nz[0] = 0;

	barrier(CLK_LOCAL_MEM_FENCE);

	if (rid >= bin) 
		return;

	int start_col_index_A, stop_col_index_A;
	int start_col_index_B, stop_col_index_B;
	// find the row
	rid = d_csrRowCReorder[ptr + rid];

	start_col_index_A = d_csrRowPtrA[rid];
	stop_col_index_A = d_csrRowPtrA[rid + 1];
	// for each column index in A
	for (i = d_csrRowPtrA[rid]; i < d_csrRowPtrA[rid + 1]; i++)
	{
		// find the corresponding row of B
		int row_id_b = d_csrColIndA[i];
		int aval = d_csrValA[i];

		start_col_index_B = d_csrRowPtrB[row_id_B];
		stop_col_index_B = d_csrRowPtrB[row_id_B + 1];

		int j;
		// for each column index of B
		for (j = start_col_index_B + tid; j < stop_col_index_B; j += local_size)
		{
			int key = d_csrColIndB[j];
			int bval = d_csrValB[j];
			int hash = (bcol * HASH_CONST) % hash_size;
			int addr = hash;
			// match and insert to the hash table
			while (true)
			{
				if (s_colhashtable[addr] == key)
				{
					atomic_add_local(&s_valhashtable[addr], aval * bval);
					break;
				}
				else if (s_colhashtable[addr] == -1)
				{
					int old = atomic_cmpxchg(&s_colhashtable[addr], -1, key);
					if (old == -1)
					{
						atomic_add_local(&s_valhashtable[addr], aval * bval);

						break;
					}
					else
					{
						hash = (hash + 1) % hash_size;
						addr = hash;
					}
				}
			}
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	// compact the results into the front of buffers
	for (i = tid; i < hash_size; i += local_size)
	{
		int index, ccol, cval;

		if (s_colhashtable[i] != -1)
		{
			index = atomic_add(s_Nz, 1);
			ccol = s_colhashtable[i];
			cval = s_valhashtable[i];
		}
		else
			index = -1;

		barrier(CLK_LOCAL_MEM_FENCE);

		if (index != -1)
		{
			s_colhashtable[index] = ccol;
			s_valhashtable[index] = cval;
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	int nnz = s_Nz[0];
	int offset = d_csrRowPtrC[rid];
	// sort and store the column index and value
	for(i = tid; i < nnz; i += local_size)
	{
		int target = s_colhashtable[i];
		int count = 0;

		for (j = 0; j < nnz; j++)
			count += (unsigned int)(s_colhashtable[j] - target) >> 31;

		d_csrColIndC[offset + count] = s_colhashtable[i];
		d_csrValC[offset + count] = s_valhashtable[i];
	}
}

/**
 * @brief Compute columns and values of C's rows with 65~2048 non-zeros
 * 
 * @param d_csrRowPtrA A's row pointer array
 * @param d_csrColIndA A's column indices array
 * @param d_csrValA  A's value array
 * @param d_csrRowPtrB B's row pointer array
 * @param d_csrColIndB B's column indices array
 * @param d_csrValB B's value array
 * @param d_csrRowPtrC C's row pointer array
 * @param d_csrColIndC C's column indices array
 * @param d_csrValC C's value array
 * @param d_csrRowCReorder Reordered C's rows
 * @param s_Nz Local memory buffer for storing the number of non-zeros
 * @param d_colhashtable Global memory hash table for column index matching
 * @param d_valhashtable Global memory hash table for value addition
 * @param bin Number of C's rows with 1 non-zero
 * @param ptr Baseline pointer for the non-zero bin
 * @param nnzSize maximum non-zeros in each bin
 */
__kernel
void compute_valC_tb_global(
        __global const int *d_csrRowPtrA,
        __global const int *d_csrColIndA,
        __global const float *d_csrValA,
        __global const int *d_csrRowPtrB,
        __global const int *d_csrColIndB,
        __global const float *d_csrValB,
        __global int *d_csrRowPtrC,
        __global int *d_csrColIndC,
        __global float *d_csrValC,
        __global const int *d_csrRowCReorder,
        __local int *s_Nz,
        __global int *d_colhashtable,
		__global float *d_colhashtable,
        const int bin,
        const int ptr,
        const int nnzSize)
{
	int rid = get_group_id(0);
	int tid = get_local_id(0);
	int local_size = get_local_size(0);
	int hash_size = 2 * nnzSize;
	int doffset = rid * hash_size;

	int i;

	if(tid == 0)
		s_Nz[0] = 0;

	if (rid >= bin)
		return;

	int start_col_index_A, stop_col_index_A;
	int start_col_index_B, stop_col_index_B;

	// find the row
	rid = d_csrRowCReorder[ptr + rid];

	start_col_index_A = d_csrRowPtrA[rid];
	stop_col_index_A = d_csrRowPtrA[rid + 1];
	// for each column index of A
	for (i = d_csrRowPtrA[rid]; i < d_csrRowPtrA[rid + 1]; i++)
	{
		// find the corresponding row of B
		int row_id_b = d_csrColIndA[i];
		int aval = d_csrValA[i];

		start_col_index_B = d_csrRowPtrB[row_id_B];
		stop_col_index_B = d_csrRowPtrB[row_id_B + 1];

		int j;
		// for each column index of B
		for (j = start_col_index_B + tid; j < stop_col_index_B; j += local_size)
		{
			// perform hash operation
			int key = d_csrColIndB[j];
			int bval = d_csrValB[j];
			int hash = (bcol * HASH_CONST) % hash_size;
			int addr = doffset + hash;
			// match and insert to the hash table
			while (true)
			{
				if (d_colhashtable[addr] == key)
				{
					atomic_add_global(&d_valhashtable[addr], aval * bval);
					break;
				}
				else if (d_colhashtable[addr] == -1)
				{
					int old = atomic_cmpxchg(&d_colhashtable[addr], -1, key);
					if (old == -1)
					{
						atomic_add_global(&d_valhashtable[addr], aval * bval);

						break;
					}
					else
					{
						hash = (hash + 1) % hash_size;
						addr = doffset + hash;
					}
				}
			}
		}
	}

	barrier(CLK_GLOBAL_MEM_FENCE);
	// Compact the results to the front
	for (i = tid; i < hash_size; i += local_size)
	{
		int index, ccol, cval;

		if (d_colhashtable[doffset + i] != -1)
		{
			index = atomic_add(s_Nz, 1);
			ccol = d_colhashtable[doffset + i];
			cval = d_valhashtable[doffset + i];
		}
		else
			index = -1;

		barrier(CLK_GLOBAL_MEM_FENCE);

		if (index != -1)
		{
			d_colhashtable[doffset + index] = ccol;
			d_valhashtable[doffset + index] = cval;
		}
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	int nnz = s_Nz[0];
	int offset = d_csrRowPtrC[rid];
	// sort and store the result
	for (i = tid; i < nnz; i += local_size)
	{
		int target = d_colhashtable[doffset + i];
		int count = 0;

		for (j = 0; j < nnz; j++)
			count += (unsigned int)(d_colhashtable[doffset + j] - target) >> 31;

		d_csrColIndC[offset + count] = d_colhashtable[doffset + i];
		d_csrValC[offset + count] = d_valhashtable[doffset + i];
	}
}