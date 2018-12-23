#define NIP_SEGMENTS 16

/**
 * @brief Reorder C's rows based on bins
 * 
 * @param d_csrRowCInnProdNum Number of products in each C's row
 * @param d_clinnPtr Base pointers for each bin
 * @param d_csrRowCReorder Buffer for storing reordered C's rows' numbers
 * @param m Number of C's rows
 */
__kernel
void compute_ReorderRowNip_kernel(
        __global const int *d_csrRowCInnProdNum,
        __global int *d_clinnPtr,
        __global int *d_csrRowCReorder,
        const int m)
{
    int global_id = get_global_id(0);

    if(global_id >= m) 
        return;
    // get the number of intermediate products of each row
    int innSize = d_csrRowPtrCinnSize[global_id];
    int location;
    // for the number of intermediate products, set the location to store in d_csrRowReorder
    if(innSize == 0) 
        location = atomic_add(&d_clinnPtr[0], 1);
    else if(innSize == 1) 
        location = atomic_add(&d_clinnPtr[1], 1);
    else if(innSize == 2) 
        location = atomic_add(&d_clinnPtr[2], 1);
    else if(8192 < innSize) 
        location = atomic_add(&d_clinnPtr[NIP_SEGMENTS - 1], 1);
    else
    {
        int i;
        
        for(i = 3; i < NIP_SEGMENTS - 1; i++)
        {
            if((1 << (i - 2)) < innSize && innSize <= (1 << (i - 1)))
            {
                location = atomic_add(&d_clinnPtr[i], 1);
                break;
            }
        }
    }
     // store the row number in the buffer   
    d_csrRowCReorder[location] = global_id;
}