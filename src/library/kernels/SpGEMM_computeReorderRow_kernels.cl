#if defined(cl_khr_global_int32_base_atomics) && defined(cl_khr_global_int32_extended_atomics)
    #pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
    #pragma OPENCL_EXTENSION cl_khr_global_int32_extended_atomics : enable
#else
    #error "Required 32-bit atomics not supported by this OpenCL implemenation."
#endif

#define INT_PROD_NUM_SEGMENTS 16

__kernel
void compute_ReorderRow_kernel(
        __global const int *d_csrRowCIntProdNum,
        __global int *d_clIntPtr,
        __global int *d_csrRowCReorder,
        const int m)
{
    int global_id = get_global_id(0);
    if(global_id >= m) return;

    int intSize = d_csrRowPtrCIntSize[global_id];
    int location;

    if(intSize == 0) location = atomic_add(&d_clIntPtr[0], 1);
    else if(intSize == 1) location = atomic_add(&d_clIntPtr[1], 1);
    else if(intSize == 2) location = atomic_add(&d_clIntPtr[2], 1);
    else if(8192 < intSize) location = atomic_add(&d_clIntPtr[INT_PROD_NUM_SEGMENTS - 1], 1);
    else
    {
        int i;
        
        for(i = 3; i < INT_PROD_NUM_SEGMENTS - 1; i++)
        {
            if((1 << (i - 2)) < intSize && intSize <= (1 << (i - 1)))
            {
                location = atomic_add(&d_clIntPtr[i], 1);
                break;
            }
        }
    }
        
    d_csrRowCReorder[location] = global_id;
}

