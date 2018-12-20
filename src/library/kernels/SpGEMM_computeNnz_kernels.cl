#define HASH_CONST 127
#define MAX_HASH_SIZE 8192

__kernel
void compute_nnzC_0(
        __global const int *d_csrRowCReorder,
        __global int *d_csrRowCNnzSize,
        const int bin)
{
    int global_id = get_global_id(0);

    if(global_id >= bin) return;

    int row_id = d_csrRowCReorder[global_id];
    d_csrRowCNnzSize[row_id] = 0;
}

__kernel
void compute_nnzC_1(
        __global const int *d_csrRowCReorder,
        __global int *d_csrRowCNnzSize,
        const int bin,
        const int ptr)
{
    int global_id = get_global_id(0);

    if(global_id >= bin) return;

    int row_id = d_csrRowCReorder[ptr + global_id];
    d_csrRowCNnzSize[row_id] = 1;
}

__kernel
void compute_nnzC_pwarp(
        __global const int *d_csrRowPtrA,
        __global const int *d_csrColIndA,
        __global const int *d_csrRowPtrB,
        __global const int *d_csrColIndB,
        __global const int *d_csrRowCReorder,
        __global int *d_csrRowCNnzSize,
        __local int *hashtable,
        const int bin,
        const int ptr,
        const int intSize)
{
    int global_id = get_global_id(0);
    int rid = global_id / intSize;

    int local_id = get_local_id(0);

    hashtable[local_id] = -1;
    
    barrier(CLK_LOCAL_MEM_FENCE);

	if (rid >= bin) return;

    int row_id = d_csrRowCReorder[ptr + rid];
    int start_col_index_A, stop_col_index_A;  // index_type
    int start_col_index_B, stop_col_index_B;  // index_type
    int tid = global_id % intSize;
    int local_rid = rid * intSize;

    start_col_index_A = d_csrRowPtrA[row_id];
    stop_col_index_A  = d_csrRowPtrA[row_id + 1];

    int i, nnz = 0;

    for (i = start_col_index_A + tid; i < stop_col_index_A; i += intSize)
    {
        int row_id_B = d_csrColIndA[i];

        start_col_index_B = d_csrRowPtrB[row_id_B];
        stop_col_index_B  = d_csrRowPtrB[row_id_B + 1];

        int j;

        for(j = start_col_index_B; j < stop_col_index_B; j++)
        {
            int key = d_csrColIndB[j];
            int hash = (bcol * HASH_CONST) % intSize;
            int addr = local_rid + hash;

            while(true)
            {
                if(hashtable[addr] == key) break;
                else if(hashtable[addr] == -1)
                {
                    int old = atomic_cmpxchg(hashtable + addr, -1, key);

                    if(old == -1)
                    {
                        nnz++;

                        break;
                    }
                }
                else
                {
                    hash = (hash + 1) % intSize;
                    addr = local_rid + hash;
                }
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if(tid == 0) hashtable[rid] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);
    atomic_add(&hashtable[rid], nnz);
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid == 0) d_csrRowPtrCNnzSize[row_id] = nnzbuffer[rid];
}

__kernel
void compute_nnzC_tb(
        __global const int *d_csrRowPtrA,
        __global const int *d_csrColIndA,
        __global const int *d_csrRowPtrB,
        __global const int *d_csrColIndB,
        __global const int *d_csrRowCReorder,
        __global int *d_csrRowPtrCNnzSize,
        __local int *hashtable,
        const int bin,
        const int ptr,
        const int intSize)
{
    int group_id = get_group_id(0);

    if(group_id >= bin) return;

    int local_id = get_local_id(0);
    int local_size = get_local_size(0);
    int i;

    for(i = local_id; i < intSize; i += local_size)
        hashtable[i] = -1;

    barrier(CLK_LOCAL_MEM_FENCE);
    
    int row_id = d_csrRowCReorder[ptr + group_id];
    int start_col_index_A = d_csrRowPtrA[row_id];
    int stop_col_index_A = d_csrRowPtrA[row_id + 1];
    int nnz = 0;

    for (i = start_col_index_A; i < stop_col_index_A; i++)
    {
        int row_id_B = d_csrColIndA[i];
        int start_col_index_B = d_csrRowPtrB[row_id_B];
        int stop_col_index_B  = d_csrRowPtrB[row_id_B + 1];
        int j;

        for(j = start_col_index_B + local_id; j < stop_col_index_B; j += local_size)
        {
            int key = d_csrColIndB[j];
            int hash = (bcol * HASH_CONST) % intSize;

            while(true)
            {
                if(hashtable[hash] == key) break;
                else if(hashtable[hash] == -1)
                {
                    int old = atomic_cmpxchg(hashtable + hash, -1, key);

                    if(old == -1)
                    {
                        nnz++;

                        break;
                    }
                }
                else hash = (hash + 1) % intSize;
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if(local_id == 0) hashtable[0] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);
    atomic_add(&hashtable[0], nnz);
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id == 0) d_csrRowPtrCNnzSize[row_id] = hashtable[0];
}

__kernel
void compute_nnzC_tb_large(
        __global const int *d_csrRowPtrA,
        __global const int *d_csrColIndA,
        __global const int *d_csrRowPtrB,
        __global const int *d_csrColIndB,
        __global const int *d_csrRowCReorder,
        __global int *d_csrRowCNnzSize,
        __global int *d_fail_count,
        __global int *d_fail_perm,
        __local int *hashtable,
        __local int *entrynum,
        const int bin,
        const int ptr)
{
    int group_id = get_group_id(0);

    if(group_id >= bin) return;

    int local_id = get_local_id(0);
    int local_size = get_local_size(0);
    int i;

    for(i = local_id; i < MAX_HASH_SIZE; i += local_size) hashtable[i] = -1;
    if(local_id == 0) entrynum[0] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    int row_id = d_csrRowCReorder[ptr + group_id];
    int start_col_index_A = d_csrRowPtrA[row_id];
    int stop_col_index_A  = d_csrRowPtrA[row_id + 1];
    int count;

    for (i = start_col_index_A + local_id; i < stop_col_index_A; i += local_size)
    {
        int row_id_B = d_csrColIndA[i];
        int start_col_index_B = d_csrRowPtrB[row_id_B];
        int stop_col_index_B  = d_csrRowPtrB[row_id_B + 1];
        int j;

        for(j = start_col_index_B; j < stop_col_index_B; j++)
        {
            int key = d_csrColIndB[j];
            int hash = (key * HASH_CONST) % (MAX_HASH_SIZE - 1);

            while(entrynum[0] < 4096)
            {
                if(hashtable[hash] == key) break;
                else if(hashtable[hash] == -1)
                {
                    int old = atomic_cmpxchg(hashtable + hash, -1, key);
                    if(old == -1)
                    {
                        atomic_add(entrynum, 1);

                        break;
                    }
                }
                else 
                    hash = (hash + 1) % (MAX_HASH_SIZE - 1);
            }
            if(entrynum[0] >= 4096)
                break;
        }
        if(entrynum[0] >= 4096)
            break;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if(entrynum[0] >= 4096)
    {
        if(local_id == 0)
        {
            int d = atomic_add(d_fail_count, 1);
            d_fail_perm[d] = row_id;
        }
    }
    else
    {
        if(local_id == 0)
            d_csrRowCNnzSize[row_id] = entrynum[0];
    }
}

__kernel
void compute_nnzC_tb_global(
        __global const int *d_csrRowPtrA,
        __global const int *d_csrColIndA,
        __global const int *d_csrRowPtrB,
        __global const int *d_csrColIndB,
        __global const int *d_fail_perm,
        __global int *d_csrRowCNnzSize,
        __global int *d_hashtable,
        const int fail_count,
        const int d_hashsize
        __global int *d_max_nnz)
{
    int group_id = get_group_id(0);

    if(group_id >= fail_count) return;

    int local_id = get_local_id(0);
    int local_size = get_local_size(0);
    int i;

    int row_id = d_fail_perm[group_id];

    int start_col_index_A = d_csrRowPtrA[row_id];
    int stop_col_index_A  = d_csrRowPtrA[row_id + 1];
    int nnz = 0;
    int addr;

    for (i = start_col_index_A + local_id; i < stop_col_index_A; i += local_size)
    {
        int row_id_B = d_csrColIndA[i];
        int start_col_index_B = d_csrRowPtrB[row_id_B];
        int stop_col_index_B  = d_csrRowPtrB[row_id_B + 1];
        int j;

        for(j = start_col_index_B; j < stop_col_index_B; j++)
        {
            int key = d_csrColIndB[j];
            int hash = (key * HASH_CONST) % d_hashsize;

            while(true)
            {
                if(hashtable[hash] == key) break;
                else if(hashtable[hash] == -1)
                {
                    int old = atomic_cmpxchg(hashtable + hash, -1, key);
                    if(old == -1)
                    {
                        nnz++;

                        break;
                    }
                }
                else 
                {
                    hash = (hash + 1) % (MAX_HASH_SIZE - 1);

                }
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if(local_id == 0) hashtable[0] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);
    atomic_add(&hashtable[0], nnz);
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id == 0)
    {
        nnz = hashtable[0];
        d_csrRowCNnzSize[row_id] = nnz;
        atomic_max(d_max_nnz, nnz);
    }
}

