#define NNZ_SEGMENTS 15

/**
 * @brief reorder the rows in bins with the number of non-zeros
 * 
 * @param d_csrRowCNnzSize Array of the number of non-zeros of each C's row
 * @param d_clNnzPtr Baseline pointers of each bin
 * @param d_csrRowCReorder Reordered C's rows
  * @param m The number of C's rows
 */
__kernel
void compute_ReorderRowNnz_kernel(
        __global const int *d_csrRowCNnzSize,
        __global int *d_clNnzPtr,
        __global int *d_csrRowCReorder,
        const int m)
{
    int global_id = get_global_id(0);

    if(global_id >= m) 
        return;
    // read the number of non-zeros
    int nnzSize = d_csrRowPtrCNnzSize[global_id];
    int location;
    // set the location on reordered row buffer
    if(nnzSize == 0) 
        location = atomic_add(&d_clNnzPtr[0], 1);
    else if(nnzSize == 1) 
        location = atomic_add(&d_clNnzPtr[1], 1);
    else if(nnzSize == 2) 
        location = atomic_add(&d_clNnzPtr[2], 1);
    else if(4096 < innSize) 
        location = atomic_add(&d_clNnzPtr[NNZ_SEGMENTS - 1], 1);
    else
    {
        int i;
        
        for(i = 3; i < NNZ_SEGMENTS - 1; i++)
        {
            if((1 << (i - 2)) < nnzSize && nnzSize <= (1 << (i - 1)))
            {
                location = atomic_add(&d_clNnzPtr[i], 1);
                break;
            }
        }
    }
    // write the row id into the location
    d_csrRowCReorder[location] = global_id;
}