#define HASH_CONST 127
#define MAX_HASH_SIZE 8192

/**
 * @brief Compute the number of nonzeros of C's rows with 0 intermediate products (to 0)
 * 
 * @param d_csrRowCReorder Reordered C's rows
 * @param d_csrRowCNnzSize Buffer for storing the number of non-zeros for each C's row (to 0)
 * @param bin Number of C's rows with 0 intermediate products
 */
__kernel
void compute_nnzC_0(
        __global const int *d_csrRowCReorder,
        __global int *d_csrRowCNnzSize,
        const int bin)
{
    int global_id = get_global_id(0);

    if(global_id >= bin) 
        return;

    // find the number of row
    int row_id = d_csrRowCReorder[global_id];
    // set the number of non-zero elements as 0
    d_csrRowCNnzSize[row_id] = 0;
}

/**
 * @brief Compute the number of nonzeros of C's rows with 1 intermediate product (to 0)
 * 
 * @param d_csrRowCReorder Reordered C's rows
 * @param d_csrRowCNnzSize Buffer for storing the number of non-zeros for each C's row (to 1)
 * @param bin Number of C's rows with 1 intermediate product
 * @param ptr Baseline pointer for reading C's rows in csrRowCReorder
 */
__kernel
void compute_nnzC_1(
        __global const int *d_csrRowCReorder,
        __global int *d_csrRowCNnzSize,
        const int bin,
        const int ptr)
{
    int global_id = get_global_id(0);

    if(global_id >= bin)
        return;
    // find the number or row
    int row_id = d_csrRowCReorder[ptr + global_id];
    // set the number of non-zero elements as 1
    d_csrRowCNnzSize[row_id] = 1;
}

/**
 * @brief Compute the number of non-zeros of C's rows with 2~64 intermediate products
 * 
 * @param d_csrRowPtrA A's row pointer array
 * @param d_csrColIndA A's column indices array
 * @param d_csrRowPtrB B's row pointer array
 * @param d_csrColIndB B's column indices array
 * @param d_csrRowCReorder Reordered C's rows
 * @param d_csrRowCNnzSize Buffer for storing the number of non-zeros for each C's row
 * @param s_hash_table Local memory buffer for matching column index
 * @param bin Number of C's rows in each bin
 * @param ptr Baseline pointer of each bin for reading C's rows
 * @param innSize Maximum number of intermediate numbers of C's rows in each bin
 * @param threads_per_row Number of threads for a column of A
 */
__kernel
void compute_nnzC_pwarp(
        __global const int *d_csrRowPtrA,
        __global const int *d_csrColIndA,
        __global const int *d_csrRowPtrB,
        __global const int *d_csrColIndB,
        __global const int *d_csrRowCReorder,
        __global int *d_csrRowCNnzSize,
        __local int *s_hash_table,
        const int bin,
        const int ptr,
        const int innSize,
        const int threads_per_row)
{
    int global_id = get_global_id(0);
    int rid = global_id / innSize;

    int local_id = get_local_id(0);
    int tid = global_id % threads_per_row;
    int local_rid = innSize * (rid % (256 / threads_per_row));
    int i, nnz = 0;

    // initialize hash table as -1
    s_hash_table[local_id] = -1;
    
    barrier(CLK_LOCAL_MEM_FENCE);

    if(rid < bin)
    {
        int rid = d_csrRowCReorder[ptr + rid];
        int start_col_index_A, stop_col_index_A;
        int start_col_index_B, stop_col_index_B;

        start_col_index_A = d_csrRowPtrA[rid];
        stop_col_index_A  = d_csrRowPtrA[rid + 1];

        // for each column index of A 
        for (i = start_col_index_A + tid; i < stop_col_index_A; i += threads_per_row)
        {
            // find the corresponding row of B
            int row_id_B = d_csrColIndA[i];

            start_col_index_B = d_csrRowPtrB[row_id_B];
            stop_col_index_B  = d_csrRowPtrB[row_id_B + 1];

            int j;
            // for each column index with threads_per_row threads
            for(j = start_col_index_B; j < stop_col_index_B; j++)
            {
                // calculate hash
                int key = d_csrColIndB[j];
                int hash = (bcol * HASH_CONST) % innSize;
                int addr = local_rid + hash;
                // match or insert the index in hash table
                while(true)
                {
                    if(s_hash_table[addr] == key) 
                        break;
                    else if(s_hash_table[addr] == -1)
                    {
                        // try to insert new index in hash table
                        int old = atomic_cmpxchg(s_hash_table + addr, -1, key);
                        // if success to insert
                        if(old == -1)
                        {
                            // increase the number of non-zeros
                            nnz++;

                            break;
                        }
                    }
                    else
                    {
                        // move to next place
                        hash = (hash + 1) % innSize;
                        addr = local_rid + hash;
                    }
                }
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // store number of non zeros for each thread in each local memory first,
    s_hash_table[local_rid + tid] = nnz;
    
    barrier(CLK_LOCAL_MEM_FENCE);

    i = threads_per_row >> 1;

    while(i > 0)
    {
        // sum up the number of non-zeros for each row
        if(tid < i)
            s_hash_table[local_rid + tid] += s_hash_table[local_rid + tid + i];

            i >>= 1;
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(tid == 0)  
    {
        // and store the number of non-zero in GLOBAL memory
        d_csrRowPtrCNnzSize[rid] = s_hash_table[local_rid];
    }
}

/**
 * @brief Compute the number of non-zeros of C's rows with 65~8192 intermediate products
 * 
 * @param d_csrRowPtrA A's row pointer array
 * @param d_csrColIndA A's column indices array
 * @param d_csrRowPtrB B's row pointer array
 * @param d_csrColIndB B's column indices array
 * @param d_csrRowCReorder Reordered C's rows
 * @param d_csrRowCNnzSize Buffer for storing the number of non-zeros for each C's row
 * @param s_hash_table Local memory buffer for matching column index
 * @param bin Number of C's rows in each bin
 * @param ptr Baseline pointer of each bin for reading C's rows
 * @param innSize Maximum number of intermediate numbers of C's rows in each bin
  * @param threads_per_row Number of threads per B's row
 */
__kernel
void compute_nnzC_tb(
        __global const int *d_csrRowPtrA,
        __global const int *d_csrColIndA,
        __global const int *d_csrRowPtrB,
        __global const int *d_csrColIndB,
        __global const int *d_csrRowCReorder,
        __global int *d_csrRowPtrCNnzSize,
        __local int *s_hash_table,
        const int bin,
        const int ptr,
        const int innSize,
        const int threads_per_row)
{
    int group_id = get_group_id(0);
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);
    int waves = local_size / threads_per_row;
    int wave_id = local_id / threads_per_row;
    int hashsize = 2 * innSize;
    int i;

    // initialize local memory buffer to -1
    for(i = local_id; i < hashSize; i += local_size)
        s_hash_table[i] = -1;

    barrier(CLK_LOCAL_MEM_FENCE);
    
    int row_id = d_csrRowCReorder[ptr + group_id];
    int start_col_index_A = d_csrRowPtrA[row_id];
    int stop_col_index_A = d_csrRowPtrA[row_id + 1];
    int nnz = 0;

    // for each column index of A
    for (i = start_col_index_A + wave_id; i < stop_col_index_A; i+= waves)
    {
        // find the corresponding for of B
        int row_id_B = d_csrColIndA[i];
        int start_col_index_B = d_csrRowPtrB[row_id_B];
        int stop_col_index_B  = d_csrRowPtrB[row_id_B + 1];
        int j;
        // for each column index of B
        for(j = start_col_index_B + local_id; j < stop_col_index_B; j += threads_per_row)
        {
            // Calculate hash
            int key = d_csrColIndB[j];
            int hash = (bcol * HASH_CONST) % hashsize;
            // match or insert to hash table
            while(true)
            {
                if(s_hash_table[hash] == key)
                    break;
                else if(s_hash_table[hash] == -1)
                {
                    // try to insert new index
                    int old = atomic_cmpxchg(s_hash_table + hash, -1, key);
                    // if success
                    if(old == -1)
                    {
                        // increase the number of non-zeros
                        nnz++;

                        break;
                    }
                }
                else
                    hash = (hash + 1) % hashsize;
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    // store the number of non-zeros into hash table
    s_hash_table[local_id] = nnz;
    
    barrier(CLK_LOCAL_MEM_FENCE);

    i = local_size << 1;
    // sum up the number of non-zeros into s_hash_table[0]
    while(i > 0)
    {
        if(local_id < i)
            s_hash_table[local_id] += s_hash_table[local_id + i];

        i >>= 1;
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0)
        d_csrRowPtrCNnzSize[row_id] = s_hash_table[0];
}

/**
 * @brief Try to compute the number of non-zeros of C's rows with 8193~ intermediate products in local 
 *        memory
 * 
 * @param d_csrRowPtrA A's row pointer array
 * @param d_csrColIndA A's column indices array
 * @param d_csrRowPtrB B's row pointer array
 * @param d_csrColIndB B's column indices array
 * @param d_csrRowCReorder Reordered C's rows
 * @param d_csrRowCNnzSize Buffer for storing the number of non-zeros for each C's row
 * @param d_fail_count Number of C's rows that failed to count the number of non-zeros
 * @param d_fail_perm Buffer for storing C's rows that failed to count the number of non-zeros
 * @param s_hash_table Local memory buffer for matching column index
 * @param entrynum Number of entries in hash table
 * @param bin Number of C's rows in the last bin
 * @param ptr Baseline pointer of the last bin for reading C's rows
 */
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
        __local int *s_hash_table,
        __local int *entrynum,
        const int bin,
        const int ptr)
{
    int group_id = get_group_id(0);
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);
    int waves = local_size / 64;
    int wave_id = local_id / 64;
    int i;

    // initialize local memory
    for(i = local_id; i < MAX_HASH_SIZE - 1; i += local_size) 
        s_hash_table[i] = -1;
    if(local_id == 0) 
        entrynum[0] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    int row_id = d_csrRowCReorder[ptr + group_id];
    int start_col_index_A = d_csrRowPtrA[row_id];
    int stop_col_index_A  = d_csrRowPtrA[row_id + 1];
    int count;
    // for each column index of A
    for (i = start_col_index_A + wave_id; i < stop_col_index_A; i += waves)
    {
        // find corresponding row of B
        int row_id_B = d_csrColIndA[i];
        int start_col_index_B = d_csrRowPtrB[row_id_B];
        int stop_col_index_B  = d_csrRowPtrB[row_id_B + 1];
        int j;
        // for each column index of B
        for(j = start_col_index_B + local_id; j < stop_col_index_B; j += 64)
        {
            int key = d_csrColIndB[j];
            int hash = (key * HASH_CONST) % (MAX_HASH_SIZE - 1);
            // match and insert index
            while(entrynum[0] < 4096)
            {
                if(s_hash_table[hash] == key) 
                    break;
                else if(s_hash_table[hash] == -1)
                {
                    int old = atomic_cmpxchg(s_hash_table + hash, -1, key);
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
        // if there are too many entries in hash table
        if(local_id == 0)
        {
            // add to failed buffer
            int d = atomic_add(d_fail_count, 1);
            d_fail_perm[d] = row_id;
        }
    }
    else
    {
        // store the non-zero value to the result buffer
        if(local_id == 0)
            d_csrRowCNnzSize[row_id] = entrynum[0];
    }
}

/**
 * @brief Compute the number of non-zeros of C's rows with 8193~ intermediate products in global memory 
 *        that was failed on local memory
 * 
 * @param d_csrRowPtrA A's row pointer array
 * @param d_csrColIndA A's column indices array
 * @param d_csrRowPtrB B's row pointer array
 * @param d_csrColIndB B's column indices array
 * @param d_fail_perm C's rows that failed to count the number of non-zeros in local memory
 * @param d_csrRowCNnzSize Buffer for storing the number of non-zeros for each C's row
 * @param d_hash_table Global memory buffer that used for counting the number of non-zeros
 * @param fail_count The number of C's rows that failed to count the number of non-zeros in local memory
 * @param d_hashsize The number of maximum intermediate products of all C's rows, size of clhashtable
 * @param s_max_buffer Local memory buffer for counting maximum non-zero elements
 * @param d_max_nnz The number of maximum non-zero elements of all C's rows
 */
__kernel
void compute_nnzC_tb_global(
        __global const int *d_csrRowPtrA,
        __global const int *d_csrColIndA,
        __global const int *d_csrRowPtrB,
        __global const int *d_csrColIndB,
        __global const int *d_fail_perm,
        __global int *d_csrRowCNnzSize,
        __global int *d_hash_table,
        const int fail_count,
        const int d_hashsize,
        __local int *s_max_buffer,
        __global int *d_max_nnz)
{
    int group_id = get_group_id(0);
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);
    int waves = local_size / 64;
    int wave_id = local_id / 64;
    int i;
    // find the row
    int row_id = d_fail_perm[group_id];

    int start_col_index_A = d_csrRowPtrA[row_id];
    int stop_col_index_A  = d_csrRowPtrA[row_id + 1];
    int nnz = 0;
    int addr;
    // for each column index of A
    for (i = start_col_index_A + wave_id; i < stop_col_index_A; i += waves)
    {
        // find the corresponding row B
        int row_id_B = d_csrColIndA[i];
        int start_col_index_B = d_csrRowPtrB[row_id_B];
        int stop_col_index_B  = d_csrRowPtrB[row_id_B + 1];
        int j;
        // for each column index of B
        for(j = start_col_index_B += local_id; j < stop_col_index_B; j += 64)
        {
            // perform hash operation
            int key = d_csrColIndB[j];
            int hash = (key * HASH_CONST) % d_hashsize;
            int addr = group_id * d_hashsize + hash;
            // match or insert the column index in global memory hash table
            while(true)
            {
                if[addr] == key)
                    break;
                else if[addr] == -1)
                {
                    // try to insert new index
                    int old = atomic_cmpxchg + addr, -1, key);

                    if(old == -1)
                    {
                        nnz++;

                        break;
                    }
                }
                else 
                {
                    hash = (hash + 1) % d_hashsize;
                    addr = group_id * d_hashsize + hash;
                }
            }
        }
    }

    barrier(CLK_GLOBAL_MEM_FENCE);
    // store the number of non-zeros counted on every thread
    s_max_buffer[local_id] = nnz;
    
    barrier(CLK_LOCAL_MEM_FENCE);

    i = local_size << 1;
    // sum up the number of non-zeros into s_hash_table[0]
    while(i > 0)
    {
        if(local_id < i)
            s_hash_table[local_id] += s_hash_table[local_id + i];

        i >>= 1;
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0)
    {
        // write to the result buffer
        nnz = s_max_buffer[0];
        d_csrRowCNnzSize[row_id] = nnz;

        atomic_max(d_max_nnz, nnz);
    }
}