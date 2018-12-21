#define HASH_CONST 127
#define MAX_HASH_SIZE 8192

__kernel
void compute_valC_1(
        __global const int *d_csrRowPtrA,
        __global const int *d_csrColIndA,
        __global const float *d_csrColValA,
        __global const int *d_csrRowPtrB,
        __global const int *d_csrColIndB,
        __global const float *d_csrColValB,
        __global int *d_csrRowPtrC,
        __global int *d_csrColIndC,
        __global float *d_csrColValC,
        __global const int *d_csrRowCReorder,
        const int bin,
        const int ptr)
{
    int global_id = get_global_id(0);

    if(global_id >= bin) 
		return;

    int row_id = d_csrRowCReorder[ptr + global_id];
    int offset = d_csrRowPtrC[row_id];
    int acol = d_csrColIndA[row_id];
    float aval = d_csrColValA[row_id];
    int brow = d_csrRowPtrB[acol];
    int bcol = d_csrColIndB[brow];
    float bval = d_csrColValB[brow];

    d_csrColIndC[offset] = bcol;
    d_csrColValC[offset] = aval * bval;
}

__kernel
void compute_valC_pwarp(
        __global const int *d_csrRowPtrA,
        __global const int *d_csrColIndA,
        __global const float *d_csrColValA,
        __global const int *d_csrRowPtrB,
        __global const int *d_csrColIndB,
        __global const float *d_csrColValB,
        __global int *d_csrRowPtrC,
        __global int *d_csrColIndC,
        __global float *d_csrColValC,
        __global const int *d_csrRowCReorder,
        __global const int *d_Nz,
        __local int *s_colhashtable,
		__local float *s_valhashtable,
        const int bin,
        const int ptr,
        const int nnzSize)
{
	int global_id = get_global_id(0);
	int rid = global_id / nnzSize;

	int local_id = get_local_id(0);

	hashtable[local_id] = -1;
	hashtable[local_id + 64] = -1;

	barrier(CLK_LOCAL_MEM_FENCE);

	if (rid >= bin) return;

	int start_col_index_A, stop_col_index_A;  // index_type
	int start_col_index_B, stop_col_index_B;  // index_type
	int tid = global_id % nnzSize;
	int hash_size = 2 * nnzSize;
	int local_rid = (rid % nnzSize) * hashSize;

	rid = d_csrRowCReorder[ptr + rid];

	start_col_index_A = d_csrRowPtrA[rid];
	stop_col_index_A = d_csrRowPtrA[rid + 1];

	int i, nnz = 0;

	if (thread_id == 0)
		d_Nz[rid] = 0;

	start_col_index_A = d_csrRowPtrA[rid];
	stop_col_index_A = d_csrRowPtrA[rid + 1];

	for (i = d_csrRowPtrA[rid] + tid; i < d_csrRowPtrA[rid + 1]; i += nnz_size)
	{
		int row_id_b = d_csrColIndA[i];
		int aval = d_csrColValA[i];

		start_col_index_B = d_csrRowPtrB[row_id_B];
		stop_col_index_B = d_csrRowPtrB[row_id_B + 1];

		int j;

		for (j = start_col_index_B + tid; j < stop_col_index_B; j++)
		{
			int key = d_csrColIndB[j];
			int bval = d_csrColValB[j];
			int hash = (bcol * HASH_CONST) & (hash_size - 1);
			int addr = local_rid + hash;
			
			while (true)
			{
				if (s_colhashtable[addr] == key)
				{
					atomic_add(&s_colhashtable[addr], aval * bval);
					break;
				}
				else if (scolhashtable[addr] == -1)
				{
					int old = atomic_cmpxchg(&s_colhashtable[addr], -1, key);
					if (old == -1)
					{
						atomic_add(&s_colhashtable[addr], aval * bval);

						break;
					}
					else
					{
						hash = (hash + 1) % hashSize;
						addr = local_rid + hash;
					}
				}
			}
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	for (i = tid; i < hashsize; i += nnzSize)
	{
		int index, ccol, cval;

		if (s_colhashtable[local_rid + i] != -1)
		{
			index = atomic_add(d_Nz + rid, 1);
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

	int nnz = d_Nz[rid];
	if (tid < nnz)
	{
		int target = s_colhashtable[local_rid + tid];
		int count = 0;

		for (j = 0; j < nnz; j++)
			count += (unsigned int)(s_colhashtable[local_rid + j] - target) >> 31;

		int offset = d_csrRowPtrC[rid];
		d_csrColIndC[offset + count] = s_colhashtable[local_rid + tid];
		d_csrColValC[offset + count] = s_valhashtable[local_rid + tid];
	}
}

__kernel
void compute_valC_tb(
        __global const int *d_csrRowPtrA,
        __global const int *d_csrColIndA,
        __global const float *d_csrColValA,
        __global const int *d_csrRowPtrB,
        __global const int *d_csrColIndB,
        __global const float *d_csrColValB,
        __global int *d_csrRowPtrC,
        __global int *d_csrColIndC,
        __global float *d_csrColValC,
        __global const int *d_csrRowCReorder,
        __global const int *d_Nz,
        __local int *s_colhashtable,
		__local float *s_valhashtable,
        const int bin,
        const int ptr,
        const int nnzSize)
{
	int rid = get_group_id(0);
	int tid = get_local_id(0);
	int local_size = get_local_size(0);
	int hash_size = 2 * nnzSize;

	int i;

	barrier(CLK_LOCAL_MEM_FENCE);

	if (rid >= bin) return;

	int start_col_index_A, stop_col_index_A;  // index_type
	int start_col_index_B, stop_col_index_B;  // index_type

	rid = d_csrRowCReorder[ptr + rid];

	start_col_index_A = d_csrRowPtrA[rid];
	stop_col_index_A = d_csrRowPtrA[rid + 1];
	
	if (thread_id == 0)
		d_Nz[rid] = 0;

	start_col_index_A = d_csrRowPtrA[rid];
	stop_col_index_A = d_csrRowPtrA[rid + 1];

	for (i = d_csrRowPtrA[rid]; i < d_csrRowPtrA[rid + 1]; i++)
	{
		int row_id_b = d_csrColIndA[i];
		int aval = d_csrColValA[i];

		start_col_index_B = d_csrRowPtrB[row_id_B];
		stop_col_index_B = d_csrRowPtrB[row_id_B + 1];

		int j;

		for (j = start_col_index_B + tid; j < stop_col_index_B; j += local_size)
		{
			int key = d_csrColIndB[j];
			int bval = d_csrColValB[j];
			int hash = (bcol * HASH_CONST) & (hash_size - 1);
			int addr = hash;

			while (true)
			{
				if (d_colhashtable[addr] == key)
				{
					atomic_add(&s_colhashtable[addr], aval * bval);
					break;
				}
				else if (s_colhashtable[addr] == -1)
				{
					int old = atomic_cmpxchg(&s_colhashtable[addr], -1, key);
					if (old == -1)
					{
						atomic_add(&s_colhashtable[addr], aval * bval);

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

	for (i = tid; i < hashsize; i += local_size)
	{
		int index, ccol, cval;

		if (s_colhashtable[i] != -1)
		{
			index = atomic_add(d_Nz + rid, 1);
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

	int nnz = d_Nz[rid];
	int offset = d_csrRowPtrC[rid];

	for(i = tid; i < nnz; i += local_size)
	{
		int target = s_colhashtable[i];
		int count = 0;

		for (j = 0; j < nnz; j++)
			count += (unsigned int)(s_colhashtable[j] - target) >> 31;

		d_csrColIndC[offset + count] = s_colhashtable[i];
		d_csrColValC[offset + count] = s_valhashtable[i];
	}
}

__kernel
void compute_valC_tb_global(
        __global const int *d_csrRowPtrA,
        __global const int *d_csrColIndA,
        __global const float *d_csrColValA,
        __global const int *d_csrRowPtrB,
        __global const int *d_csrColIndB,
        __global const float *d_csrColValB,
        __global int *d_csrRowPtrC,
        __global int *d_csrColIndC,
        __global float *d_csrColValC,
        __global const int *d_csrRowCReorder,
        __global const int *d_Nz,
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
	int doffset = rid * hashsize;

	int i;

	if (rid >= bin) return;

	int start_col_index_A, stop_col_index_A;  // index_type
	int start_col_index_B, stop_col_index_B;  // index_type

	rid = d_csrRowCReorder[ptr + rid];

	start_col_index_A = d_csrRowPtrA[rid];
	stop_col_index_A = d_csrRowPtrA[rid + 1];

	if (thread_id == 0)
		d_Nz[rid] = 0;

	start_col_index_A = d_csrRowPtrA[rid];
	stop_col_index_A = d_csrRowPtrA[rid + 1];

	for (i = d_csrRowPtrA[rid]; i < d_csrRowPtrA[rid + 1]; i++)
	{
		int row_id_b = d_csrColIndA[i];
		int aval = d_csrColValA[i];

		start_col_index_B = d_csrRowPtrB[row_id_B];
		stop_col_index_B = d_csrRowPtrB[row_id_B + 1];

		int j;

		for (j = start_col_index_B + tid; j < stop_col_index_B; j += local_size)
		{
			int key = d_csrColIndB[j];
			int bval = d_csrColValB[j];
			int hash = (bcol * HASH_CONST) & (hash_size - 1);
			int addr = doffset + hash;

			while (true)
			{
				if (d_colhashtable[addr] == key)
				{
					atomic_add(&d_colhashtable[addr], aval * bval);
					break;
				}
				else if (d_colhashtable[addr] == -1)
				{
					int old = atomic_cmpxchg(&d_colhashtable[addr], -1, key);
					if (old == -1)
					{
						atomic_add(&d_colhashtable[addr], aval * bval);

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

	for (i = tid; i < hashsize; i += local_size)
	{
		int index, ccol, cval;

		if (d_colhashtable[doffset + i] != -1)
		{
			index = atomic_add(d_Nz + rid, 1);
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

	int nnz = d_Nz[rid];
	int offset = d_csrRowPtrC[rid];

	for (i = tid; i < nnz; i += local_size)
	{
		int target = d_colhashtable[doffset + i];
		int count = 0;

		for (j = 0; j < nnz; j++)
			count += (unsigned int)(d_colhashtable[doffset + j] - target) >> 31;

		d_csrColIndC[offset + count] = d_colhashtable[doffset + i];
		d_csrColValC[offset + count] = d_valhashtable[doffset + i];
	}
}

