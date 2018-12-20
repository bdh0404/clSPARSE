#define HASH_CONST 127
#define MAX_HASH_SIZE 8192

__kernel
void compute_nnzC_1(
        __global const int *d_csrRowPtrA,
        __global const int *d_csrColIndA,
        __global const int *d_csrValA,
        __global const int *d_csrRowPtrB,
        __global const int *d_csrColIndB,
        __global const int *d_csrValB,
        __global const int *d_csrRowCReorder,
        __global const int *d_csrColIndC,
        __global const int *d_csrValC,
        const int bin)
{
    int global_id = get_global_id(0);

    if(global_id >= bin) return;

    int row_id = d_csrRowCReorder[global_id];
    int col_index_A = d_csrRowPtrA[row_id];
    int row_id_B = d_csrColIndA[col_index_A];
    int val_A = d_csrValA[col_index_A];
    col_index_B = d_csrRowPtrB[row_id_B];
    int val_B = d_csrValB[col_index_B];
    int col_index_C = d_csrRowPtrC[row_id];
    d_csrColIndC[col_index_C] = col_index_B;
    d_csrValC = val_A * val_B;
}

__kernel
void compute_nnzC_pwarp(
        __global const int *d_csrRowPtrA,
        __global const int *d_csrColIndA,
        __global const int *d_csrRowPtrB,
        __global const int *d_csrColIndB,
        __global const int *d_csrRowCReorder,
        __global int *d_csrRowPtrCNnzSize,
        __local *hashtable,
        const int bin,
        const int ptr,
        const int intSize)
{
    int global_id = get_global_id(0);
    int rid = global_id / intSize;

    if (rid >= bin) return;

    int local_id = get_local_id(0);

    hashtable[local_id] = -1;
    
    barrier(CLK_LOCAL_MEM_FENCE);

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
