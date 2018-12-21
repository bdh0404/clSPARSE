#if defined(cl_khr_global_int32_base_atomics) && defined(cl_khr_global_int32_extended_atomics)
    #pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
    #pragma OPENCL_EXTENSION cl_khr_global_int32_extended_atomics : enable
#else
    #error "Required 32-bit atomics not supported by this OpenCL implemenation."
#endif

#define NNZ_SEGMENTS 15

__kernel
void compute_ReorderRowNnz_kernel(
        __global const int *d_csrRowCNnzSize,
        __global int *d_clNnzPtr,
        __global int *d_csrRowCReorder,
        const int m)
{
    int global_id = get_global_id(0);
    if(global_id >= m) return;

    int nnzSize = d_csrRowPtrCNnzSize[global_id];
    int location;

    if(nnzSize == 0) location = atomic_add(&d_clNnzPtr[0], 1);
    else if(nnzSize == 1) location = atomic_add(&d_clNnzPtr[1], 1);
    else if(nnzSize == 2) location = atomic_add(&d_clNnzPtr[2], 1);
    else if(4096 < innSize) location = atomic_add(&d_clNnzPtr[NNZ_SEGMENTS - 1], 1);
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
        
    d_csrRowCReorder[location] = global_id;
}