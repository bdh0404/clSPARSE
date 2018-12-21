#if defined(cl_khr_global_int32_base_atomics) && defined(cl_khr_global_int32_extended_atomics)
    #pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
    #pragma OPENCL_EXTENSION cl_khr_global_int32_extended_atomics : enable
#else
    #error "Required 32-bit atomics not supported by this OpenCL implemenation."
#endif

#define NIP_SEGMENTS 16

__kernel
void compute_ReorderRowNip_kernel(
        __global const int *d_csrRowCInnProdNum,
        __global int *d_clinnPtr,
        __global int *d_csrRowCReorder,
        const int m)
{
    int global_id = get_global_id(0);
    if(global_id >= m) return;

    int innSize = d_csrRowPtrCinnSize[global_id];
    int location;

    if(innSize == 0) location = atomic_add(&d_clinnPtr[0], 1);
    else if(innSize == 1) location = atomic_add(&d_clinnPtr[1], 1);
    else if(innSize == 2) location = atomic_add(&d_clinnPtr[2], 1);
    else if(8192 < innSize) location = atomic_add(&d_clinnPtr[NIP_SEGMENTS - 1], 1);
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
        
    d_csrRowCReorder[location] = global_id;
}