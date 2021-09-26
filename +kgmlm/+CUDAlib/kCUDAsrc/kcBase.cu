/*
 * kcBase.cu
 * classes for storing matrices (including stacks of matrices)
 * keeps track of all the GPU stuff
 *
 * Package GMLM_dmc for dimensionality reduction of neural data.
 *   
 *  References
 *   Kenneth Latimer & David Freeedman (2021). Low-dimensional encoding of 
 *   decisions in parietal cortex reflects long-term training history.
 *   bioRxiv
 *
 *  Copyright (c) 2021 Kenneth Latimer
 *
 *   This software is distributed under the GNU General Public
 *   License (version 3 or later); please refer to the file
 *   License.txt, included with the software, for details.
 */
#include "kcBase.hpp"

namespace kCUDA {
    

template <typename FPTYPE> 
__global__ void GPUData_kernel_assign(GPUData_kernel<FPTYPE> * gd, FPTYPE * data, size_t x, size_t y, size_t z, size_t ld, size_t inc) {
    size_t row   = blockIdx.x * blockDim.x + threadIdx.x;
    if(row == 0) {
        gd->data = data;
        gd->x = x;
        gd->y = y;
        gd->z = z;

        gd->ld = ld;
        gd->inc = inc;
    }
}
//=============================================================================================================================================================
//=============================================================================================================================================================
//=============================================================================================================================================================
// here we define the methods for the GPUData class
        

// constructor - sets everything to blank
template <class FPTYPE>
GPUData<FPTYPE>::GPUData() {
    data_size = make_cudaExtent(0, 0, 0);
    data_size_bytes = make_cudaExtent(0, 0, 0);

    ld_host  = 0;
    ld_gpu   = 0;
    inc_host = 0;
    inc_gpu  = 0;

    allocated_gpu  = false;
    allocated_host = false;
    is_stacked_gpu = false;

    data_gpu.ptr = NULL;
    data_host    = NULL;
    data_kernel  = NULL;
    devNum = -1;
}

    // constructor - allocates GPU memory with given size, returns a CUDA status
template <class FPTYPE>
GPUData<FPTYPE>::GPUData(cudaError_t & ce, GPUData_HOST_ALLOCATION include_host, const cudaStream_t stream, size_t width, size_t height, size_t depth, bool stacked_depth) : GPUData<FPTYPE>() {
    ce = allocate_gpu(include_host, stream, width, height, depth, stacked_depth);
}
template <class FPTYPE>
GPUData<FPTYPE>::GPUData(cudaError_t & ce, GPUData_HOST_ALLOCATION include_host, const cudaStream_t stream, cudaExtent size, bool stacked_depth) {
    GPUData(ce, include_host, stream, size.width, size.height, size.depth, stacked_depth);
}


template <class FPTYPE>
GPUData<FPTYPE>::~GPUData() {
    deallocate();
}

//allocates GPU memory (and page-locked host memory if requested)
template <class FPTYPE>
cudaError_t GPUData<FPTYPE>::allocate_gpu(GPUData_HOST_ALLOCATION include_host, const cudaStream_t stream, size_t width, size_t height, size_t depth, bool stacked_depth) {
    cudaError_t ce = cudaSuccess;
    // deallocate any existing memory
    if(isAllocated()) {
        ce = deallocate();
        if(ce != cudaSuccess) {
            return ce;
        }
    }

    //setup dimensions
    data_size       = make_cudaExtent(width,                  height, depth);
    data_size_bytes = make_cudaExtent(width * sizeof(FPTYPE), height, depth);
    data_size_c     = make_cudaExtent(width,                  height, depth);

    data_gpu.ptr = NULL;
    data_host    = NULL;
    data_kernel  = NULL;
    allocated_host = false;
    allocated_gpu  = false;
    page_locked    = false;

    //allocate GPU memory
    ce = cudaGetDevice(&devNum);
    
    stacked_depth = stacked_depth || depth == 1;


    if(!stacked_depth) {
        // do not stack depth dimension into a matrix
        ld_host = getSize(0);
        inc_host = (depth <= 1) ? 0 : ld_host * height;
        ld_gpu  = 0;
        inc_gpu = 0;

        size_t total_size = ld_host * getSize(1)  * getSize(2);
        if(ce == cudaSuccess) {
            if(size() > 0) {
                ce = cudaMalloc3D(&(data_gpu), data_size_bytes);

                ld_gpu  = data_gpu.pitch / sizeof(FPTYPE);
                inc_gpu = (depth <= 1) ? 0 : ld_gpu * height;
            }
            else {
                data_gpu = make_cudaPitchedPtr(NULL, 0, 0, height) ;
                data_gpu.ptr = NULL;
                ld_gpu  = 0;
                inc_gpu = 0;
            }
        }
        is_stacked_gpu = false;

    }
    else {
        // stack depth dimension into a matrix
        ld_host = getSize(0) * getSize(2);
        inc_host = (depth <= 1) ? 0 : getSize(0);
        ld_gpu  = 0;
        inc_gpu = 0;

        if(ce == cudaSuccess) {
            if(size() > 0) {
                ce = cudaMallocPitch(&(data_gpu.ptr), &(data_gpu.pitch), getSize(0) * getSize(2) * sizeof(FPTYPE), getSize(1)); 
                
                ld_gpu  = data_gpu.pitch / sizeof(FPTYPE);
                inc_gpu = (depth <= 1) ? 0 : getSize(0);
            }
            else {
                data_gpu = make_cudaPitchedPtr(NULL, 0, 0, height) ;
                data_gpu.ptr = NULL;
                ld_gpu  = 0;
                inc_gpu = 0;
            }
        }
        is_stacked_gpu = true;
    }
        
    //page locked memory if requested
    if(include_host == GPUData_HOST_PAGELOCKED && ce == cudaSuccess) {
        if(size() > 0) {
            ce = cudaMallocHost(reinterpret_cast<void**>(&(data_host)), size() * sizeof(FPTYPE));
        }   
        else {
            data_host = NULL;
        }
        page_locked = true;
        allocated_host = true;
    }
    else if(include_host == GPUData_HOST_STANDARD && ce == cudaSuccess) {
        if(size() > 0) {
            data_host = new FPTYPE[size()];
        }   
        else {
            data_host = NULL;
        }
        page_locked = false;
        allocated_host = true;
    }
    //puts pointer and size info on GPU
    if(ce == cudaSuccess) {
        ce = cudaMalloc(reinterpret_cast<void**>(&(data_kernel)), sizeof(GPUData_kernel<FPTYPE>));
        if(ce == cudaSuccess) {
            GPUData_kernel_assign<<<1,1,0,stream>>>(data_kernel, reinterpret_cast<FPTYPE*>(data_gpu.ptr), getSize(0), getSize(1), getSize(2), ld_gpu, inc_gpu); 
            ce = cudaGetLastError();
        }
        allocated_gpu = true;
    }
    
    inc_gpu_bytes  = inc_gpu  * sizeof(FPTYPE);
    inc_host_bytes = inc_host * sizeof(FPTYPE);
    ld_gpu_bytes   = ld_gpu  * sizeof(FPTYPE);
    ld_host_bytes  = ld_host * sizeof(FPTYPE);
    return ce;
}

//resizes current data (within pre-allocated bounds)
template <class FPTYPE>
cudaError_t GPUData<FPTYPE>::resize(const cudaStream_t stream, int width, int height, int depth) {
    cudaError_t ce = cudaSuccess;
    
    //default values
    width  = (width  < 0) ? data_size_c.width  : width;
    height = (height < 0) ? data_size_c.height : height;
    depth  = (depth  < 0) ? data_size_c.depth  : depth;
    
    //if invalid
    if(width > data_size.width || height > data_size.height || depth > data_size.depth) {
        ce = cudaErrorInvalidValue;
    }

    if(width != data_size_c.width || data_size_c.height != height || data_size_c.depth != depth) { 
        if(depth <= 1) {
            inc_gpu = 0;
            inc_gpu_bytes = 0;   
        }
        else if(!is_stacked_gpu && depth != data_size_c.depth) {
            inc_gpu = data_size_c.height * ld_gpu;
            inc_gpu_bytes = sizeof(FPTYPE) * inc_gpu;       
        }
        else if(is_stacked_gpu && width != data_size_c.width) {
            inc_gpu = width;
            inc_gpu_bytes = sizeof(FPTYPE) * width;        
        }
        data_size_c.width  = width;
        data_size_c.height = height;
        data_size_c.depth  = depth;

        //reset values
        if(isOnGPU() && ce == cudaSuccess) {
            ce = cudaSetDevice(devNum);

            if(ce == cudaSuccess) {
                GPUData_kernel_assign<<<1,1,0,stream>>>(data_kernel, reinterpret_cast<FPTYPE*>(data_gpu.ptr), getSize(0), getSize(1), getSize(2), ld_gpu, inc_gpu); 
                ce = cudaGetLastError();
            }
        }
    }

    return ce;
}

//allocates GPU memory (is page-locked host memory if requested, otherwise regular host memory)
template <class FPTYPE>
cudaError_t GPUData<FPTYPE>::allocate_host(bool page_locked_memory, size_t width, size_t height, size_t depth) {
    cudaError_t ce;
    // deallocate any existing memory
    if(isAllocated()) {
        ce = deallocate();
        if(ce != cudaSuccess) {
            return ce;
        }
    }

    //setup dimensions
    data_size       = make_cudaExtent(width, height, depth);
    data_size_bytes = make_cudaExtent(width * sizeof(FPTYPE), height, depth);
    data_size_c     = make_cudaExtent(width,                  height, depth);

    //allocate GPU memory
    data_gpu.ptr  = NULL;
    data_host     = NULL;
    data_kernel   = NULL;

    ld_host = getSize(0);
    inc_gpu = 0;
    ld_gpu  = 0;
    inc_host = (depth <= 1) ? 0 : ld_host * height;

    //allocate memory
    if(size() > 0) {
        if(page_locked_memory) {
            ce = cudaMallocHost(reinterpret_cast<void**>(&(data_host)), size() * sizeof(FPTYPE));
            cudaGetDevice(&devNum); 
            page_locked = true;
        }
        else {
            ce = cudaSuccess;
            data_host = new FPTYPE[size()];
            devNum = -1;
            page_locked = false;
        }
    }
    else {
        if(page_locked_memory) {
            data_host = NULL;
            page_locked = true;
        }
        else {
            data_host = NULL;
            devNum = -1;
        }
    }
    allocated_host = true;
    allocated_gpu = false;
    inc_gpu_bytes  = inc_gpu  * sizeof(FPTYPE);
    inc_host_bytes = inc_host * sizeof(FPTYPE);
    ld_gpu_bytes   = ld_gpu   * sizeof(FPTYPE);
    ld_host_bytes  = ld_host  * sizeof(FPTYPE);
    return ce;
}

//deallocates CUDA memory
template <class FPTYPE>
cudaError_t GPUData<FPTYPE>::deallocate() {
    cudaError_t ce = cudaSuccess;
    if(isAllocated()) {
        ce = cudaSetDevice(devNum);
        if(data_host != NULL && !page_locked && allocated_host) {
            delete[] data_host;
        }
        if(ce == cudaSuccess && data_host != NULL && page_locked && allocated_host) {
            ce = cudaFreeHost(data_host);
        }
        if(ce == cudaSuccess && data_gpu.ptr != NULL && allocated_gpu) {
            ce = cudaFree(data_gpu.ptr);
        }
        data_gpu.ptr = NULL;
        data_host = NULL;
        allocated_gpu = false;
        allocated_host = false;
        page_locked = false;
    }
    if(ce == cudaSuccess && data_kernel != NULL) {
        ce = cudaFree(data_kernel);
    }
    data_kernel = NULL;
    return ce;
}

template <typename FPTYPE> 
__global__ void GPUData_3DCopy(GPUData_kernel<FPTYPE> * dest, const GPUData_kernel<FPTYPE> * source, const cudaPos copyPos_dest) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if(x < source->x && y < source->y && z < source->z) {
        (*dest)(x + copyPos_dest.x, y + copyPos_dest.y, z + copyPos_dest.z) = (*source)(x, y, z);
    }
}


//copy a GPU array into current GPU array -> must be same size!
template <class FPTYPE>
cudaError_t GPUData<FPTYPE>::copyTo(const cudaStream_t stream, const GPUData<FPTYPE> * source, bool allowSmaller, const cudaPos copyPos_dest) {
    //if both valid
    if(source == NULL || !(source->isAllocated()) || !isAllocated()) {
        return cudaErrorInvalidValue;
    }

    //check matching sizes!
    if((!allowSmaller && getSize(0) != source->getSize(0) + copyPos_dest.x) || (allowSmaller && getSize(0) < source->getSize(0) + copyPos_dest.x)) {
        return cudaErrorInvalidValue;
    }
    if((!allowSmaller && getSize(1) != source->getSize(1) + copyPos_dest.y) || (allowSmaller && getSize(1) < source->getSize(1) + copyPos_dest.y)) {
        return cudaErrorInvalidValue;
    }
    if((!allowSmaller && getSize(2) != source->getSize(2) + copyPos_dest.z) || (allowSmaller && getSize(2) < source->getSize(2) + copyPos_dest.z)) {
        return cudaErrorInvalidValue;
    }

    //if source host allocated
    cudaError_t ce = cudaSuccess;
    if(source->isOnHost())  {

        //if dest host allocated
        if(isOnHost()) {
            for(int xx_c = 0; xx_c < source->getSize(0); xx_c++) {
                int xx = xx_c + copyPos_dest.x;
                for(int yy_c = 0; yy_c < source->getSize(1); yy_c++) {
                    int yy = yy_c + copyPos_dest.y;
                    for(int zz_c = 0; zz_c < source->getSize(2); zz_c++) {
                        int zz = zz_c + copyPos_dest.z;
//                         data_host(xx, yy, zz) = source(xx, yy, zz);
                        data_host[xx + yy*ld_host + zz*inc_host] = source->data_host[xx + yy*source->ld_host + zz*source->inc_host];
                    }
                }
            }
            //if dest GPU memory
            if(isOnGPU())  {
                return copyHostToGPU(stream);
            }
        }

        //if dest GPU memory
        else if(isOnGPU()) {
            //set current GPU
            if(size() > 0) {
                ce = cudaSetDevice(devNum);
                if(ce == cudaSuccess) {
                    for(int zz = 0; zz < source->getSize(2) && ce == cudaSuccess; zz++) {
                        ce = cudaMemcpy2DAsync(getData_gpu() + copyPos_dest.x + copyPos_dest.y * ld_gpu + ( copyPos_dest.z + zz) * inc_gpu,
                             ld_gpu_bytes,
                             source->data_host + zz * source->inc_host,
                             source->ld_host_bytes,
                             source->data_size_bytes.width, 
                             source->getSize(1),
                             cudaMemcpyHostToDevice, stream);
                    }
                }
            }   
        }
    }
    // else if source is on GPU
    else if(source->isOnGPU()) {
        if(source->devNum == devNum) {
            //set current GPU
            if(size() > 0) {
                ce = cudaSetDevice(devNum);
                if(ce == cudaSuccess) {
                    dim3 block_size;
                    block_size.x = min(static_cast<size_t>(1024), source->getSize(0));
                    block_size.y = 1;
                    block_size.z = 1;
                    dim3 grid_size;
                    grid_size.x = source->getSize(0) / block_size.x + ((source->getSize(0) % block_size.x == 0)? 0:1);
                    grid_size.y = source->getSize(1) / block_size.y + ((source->getSize(1) % block_size.y == 0)? 0:1);
                    grid_size.z = source->getSize(2) / block_size.z + ((source->getSize(2) % block_size.z == 0)? 0:1);
                    GPUData_3DCopy<<<grid_size, block_size,  0, stream>>>(device(), source->device(), copyPos_dest);
                    ce = cudaGetLastError();
                }
            }   
        }
        else {
            return cudaErrorInvalidValue;
            // NOT SUPPORTED YET!
        }
    }
    return ce;
}



//copy a GPU array into current GPU array -> must be same size!
template <class FPTYPE>
cudaError_t GPUData<FPTYPE>::copyTo(const cudaStream_t stream, const GLData<FPTYPE> * source, bool allowSmaller, const cudaPos copyPos_dest) {
    //if both valid
    if(source == NULL || !isAllocated()) {
        return cudaErrorInvalidValue;
    }

    //check matching sizes!
    if((!allowSmaller && getSize(0) != source->getSize(0) + copyPos_dest.x) || (allowSmaller && getSize(0) < source->getSize(0) + copyPos_dest.x)) {
        return cudaErrorInvalidValue;
    }
    if((!allowSmaller && getSize(1) != source->getSize(1) + copyPos_dest.y) || (allowSmaller && getSize(1) < source->getSize(1) + copyPos_dest.y)) {
        return cudaErrorInvalidValue;
    }
    if((!allowSmaller && getSize(2) != source->getSize(2) + copyPos_dest.z) || (allowSmaller && getSize(2) < source->getSize(2) + copyPos_dest.z)) {
        return cudaErrorInvalidValue;
    }

    //if source host allocated
    cudaError_t ce = cudaSuccess;

    //if dest host allocated
    if(isOnHost()) {
        for(int xx_c = 0; xx_c < source->getSize(0); xx_c++) {
            int xx = xx_c + copyPos_dest.x;
            for(int yy_c = 0; yy_c < source->getSize(1); yy_c++) {
                int yy = yy_c + copyPos_dest.y;
                for(int zz_c = 0; zz_c < source->getSize(2); zz_c++) {
                    int zz = zz_c + copyPos_dest.z;
//                         data_host(xx, yy, zz) = source(xx, yy, zz);
                    data_host[xx + yy*ld_host + zz*inc_host] = source->getData()[xx + yy*source->getLD() + zz*source->getInc()];
                }
            }
        }
        //if dest GPU memory
        if(isOnGPU())  {
            return copyHostToGPU(stream);
        }
    }

    //if dest GPU memory
    else if(isOnGPU()) {
        //set current GPU
        if(size() > 0) {
            ce = cudaSetDevice(devNum);
            if(ce == cudaSuccess) {
                for(int zz = 0; zz < source->getSize(2) && ce == cudaSuccess; zz++) {
                    ce = cudaMemcpy2DAsync(getData_gpu() + (copyPos_dest.z + zz) * inc_gpu + copyPos_dest.y * ld_gpu + copyPos_dest.x,
                         ld_gpu_bytes,
                         source->getData() + zz * source->getInc(),
                         source->getLD() * sizeof(FPTYPE),
                         source->getSize(0) * sizeof(FPTYPE), 
                         source->getSize(1),
                         cudaMemcpyHostToDevice, stream);
                }
            }
        }   
    }

    return ce;
}

//copies data in GPU memory to host (if both as assigned)
template <class FPTYPE>
cudaError_t GPUData<FPTYPE>::copyGPUToHost(const cudaStream_t stream) {
    if(allocated_host && allocated_gpu) {
        if(size() > 0) {
            //set current GPU
            cudaError_t ce = cudaSetDevice(devNum);
            //for each z (this loop is easier to me than setting up that cursed memcpy3d call)
            for(int zz = 0; zz < getSize(2) && ce == cudaSuccess; zz++) {
                ce = cudaMemcpy2DAsync(getData_host() + zz * inc_host,
                     ld_host_bytes,
                     getData_gpu() + zz * inc_gpu,
                     data_gpu.pitch,
                     getSize(0) * sizeof(FPTYPE), 
                     getSize(1),
                     cudaMemcpyDeviceToHost, stream);
            }
            return ce;
        }
        else {
            return cudaSuccess;
        }
    }
    else {
        return cudaErrorInvalidValue;
    }
}

//copies data in host memory to GPU (if both as assigned)
template <class FPTYPE>
cudaError_t GPUData<FPTYPE>::copyHostToGPU(const cudaStream_t stream) {
    if(isOnHost() && isOnGPU()) {
        //set current GPU
        cudaError_t ce = cudaSetDevice(devNum); 
        //for each z
        for(int zz = 0; zz < getSize(2) && ce == cudaSuccess; zz++) {
            ce = cudaMemcpy2DAsync(getData_gpu() + zz * inc_gpu,
                 ld_gpu_bytes,
                 getData_host() + zz * inc_host,
                 ld_host_bytes,
                 getSize(0) * sizeof(FPTYPE), 
                 getSize(1), 
                 cudaMemcpyHostToDevice, stream);
        }
        return ce;
    }
    else {
        return cudaErrorInvalidValue;
    }
}


template <typename FPTYPE> 
__global__ void GPUData_array_kernel_assign(GPUData_array_kernel<FPTYPE> * gd, GPUData_kernel<FPTYPE> ** data, size_t N) {
    size_t row   = blockIdx.x * blockDim.x + threadIdx.x;
    if(row == 0) {
        gd->data = data;
        gd->N = N;
    }
}

template <class FPTYPE>
GPUData_array<FPTYPE>::GPUData_array() {
    data_gpu = NULL;
    data_kernel = NULL;
    N_elements = 0;

    devNum = -1;
}
template <class FPTYPE>
GPUData_array<FPTYPE>::GPUData_array(cudaError_t & ce, std::vector<GPUData<FPTYPE> *> & data, const cudaStream_t stream, std::shared_ptr<GPUGL_msg> msg) {
    data_gpu = NULL;
    data_kernel = NULL;
    N_elements = 0;

    devNum = -1;

    ce = allocate(data, stream, msg);
}
        
// destructor destroys memory - does not check for CUDA errors! Recommend calling deallocate manually
template <class FPTYPE>
GPUData_array<FPTYPE>::~GPUData_array() {
    if(isAllocated()) {
        deallocate();
    }
}
        
        
// deallocate everything
template <class FPTYPE>
cudaError_t GPUData_array<FPTYPE>::deallocate() {
    cudaError_t ce = cudaSuccess;
    if(data_gpu != NULL || data_kernel != NULL) {
        ce = cudaSetDevice(devNum);
    }
    if(ce == cudaSuccess && data_gpu != NULL) {
        ce = cudaFree(data_gpu);
        data_gpu = NULL;
    }
    if(ce == cudaSuccess && data_kernel != NULL) {
        ce = cudaFree(data_kernel);
        data_kernel = NULL;
    }
    N_elements = 0;
    devNum = -1;
    return ce;
}
        
template <class FPTYPE>
cudaError_t GPUData_array<FPTYPE>::allocate(std::vector<GPUData<FPTYPE> *> & data, const cudaStream_t stream, std::shared_ptr<GPUGL_msg> msg) {
    cudaError_t ce = cudaSuccess;
    if(isAllocated()) {
        ce = deallocate();
        if(ce != cudaSuccess) {
            return ce;
        }
    }

    N_elements = data.size();
    if(N_elements == 0) {
        std::ostringstream output_stream;
        output_stream << "GPUData_array errors: no data given!";
        msg->callErrMsgTxt(output_stream);
    }

    data_host.assign(N_elements, NULL);
    for(int ss = 0; ss < N_elements; ss++) {
        //make sure each var is on GPU
        if(!data[ss]->isOnGPU()) {
            std::ostringstream output_stream;
            output_stream << "GPUData_array errors: data not on GPU!";
            msg->callErrMsgTxt(output_stream);
        }

        //make sure is on CORRECT gpu
        if(ss == 0) {
            devNum = data[ss]->getDevice();
        }
        else {
            if(data[ss]->getDevice() != devNum) {
                std::ostringstream output_stream;
                output_stream << "GPUData_array errors: data must all be on same GPU!";
                msg->callErrMsgTxt(output_stream);
            }
        }

        //add ptr
        data_host[ss] = data[ss]->device();
    }

    //make sure correct device is set
    if(ce == cudaSuccess) {
        ce = cudaSetDevice(devNum);
    }

    //allocate GPU space
    if(ce == cudaSuccess) {
        ce = cudaMalloc(reinterpret_cast<void**>(&(data_gpu)), data_host.size() * sizeof(GPUData_kernel<FPTYPE> *));
    }

    //copy vectors to GPU
    if(ce == cudaSuccess) {
        ce = cudaMemcpyAsync(data_gpu, data_host.data(), data_host.size() * sizeof(GPUData_kernel<FPTYPE> *), cudaMemcpyHostToDevice, stream);
    }

    //assign GPU object
    if(ce == cudaSuccess) {
        cudaMalloc(reinterpret_cast<void**>(&(data_kernel)), sizeof(GPUData_array_kernel<FPTYPE> *) );

        GPUData_array_kernel_assign<<<1,1,0,stream>>>(data_kernel, data_gpu, N_elements); 
        ce = cudaGetLastError();
    }

    return ce;
}

/* kernel for a quick MM operation X*B where X is tall * skinny and B is small
*     Trying to get a speedup from CUBLAS in regions where its slow.
*/

template <class FPTYPE>
__global__ void kernel_MM_quick(GPUData_kernel<FPTYPE> * XF, const GPUData_kernel<FPTYPE> * X, const GPUData_kernel<FPTYPE> * F, const FPTYPE alpha, const FPTYPE beta, const cublasOperation_t op_A, const cublasOperation_t op_B)   {
    int rr_start = blockIdx.y * blockDim.y + threadIdx.y;
    size_t row   = blockIdx.x * blockDim.x + threadIdx.x;
    size_t A     = blockIdx.z * blockDim.z + threadIdx.z;
    if(row < XF->x && A < X->z) {
        if(op_A == CUBLAS_OP_N && op_B == CUBLAS_OP_N) {
            for(int rr = rr_start; rr < XF->y; rr+= blockDim.y * gridDim.y) {
                FPTYPE ll = 0;
                for(int tt = 0; tt < F->x; tt++) {
                    ll += (*X)(row, tt, A) * (*F)(tt, rr, A);
                }
                (*XF)(row, rr, A) = alpha*ll + beta*(*XF)(row, rr, A);
            }
        }
        else if(op_A == CUBLAS_OP_N) {
            for(int rr = rr_start; rr < XF->y; rr+= blockDim.y * gridDim.y) {
                FPTYPE ll = 0;
                for(int tt = 0; tt < F->y; tt++) {
                    ll += (*X)(row, tt, A) * (*F)(rr, tt, A);
                }
                (*XF)(row, rr, A) = alpha*ll + beta*(*XF)(row, rr, A);
            }
        }
        else if(op_B == CUBLAS_OP_N) {
            for(int rr = rr_start; rr < XF->y; rr+= blockDim.y * gridDim.y) {
                FPTYPE ll = 0;
                for(int tt = 0; tt < F->x; tt++) {
                    ll += (*X)(tt, row, A) * (*F)(tt, rr, A);
                }
                (*XF)(row, rr, A) = alpha*ll + beta*(*XF)(row, rr, A);
            }
        }
        else {
            for(int rr = rr_start; rr < XF->y; rr+= blockDim.y * gridDim.y) {
                FPTYPE ll = 0;
                for(int tt = 0; tt < F->y; tt++) {
                    ll += (*X)(tt, row, A) * (*F)(rr, tt, A);
                }
                (*XF)(row, rr, A) = alpha*ll + beta*(*XF)(row, rr, A);
            }
        }
    }
}



template <class FPTYPE>
cublasStatus_t GPUData<FPTYPE>::GEMM(GPUData<FPTYPE> * C, const GPUData<FPTYPE> * B, const cublasHandle_t handle, const cublasOperation_t op_A, const cublasOperation_t op_B, const FPTYPE alpha, const FPTYPE beta) {
    
    cublasStatus_t ce = CUBLAS_STATUS_SUCCESS;

    size_t depth_A = getSize(2);
    size_t depth_B = B->getSize(2);
    size_t depth_C = C->getSize(2);
    size_t rows_A = getSize(0);
    size_t rows_B = B->getSize(0);
    size_t rows_C = C->getSize(0);
    size_t cols_A = getSize(1);
    size_t cols_B = B->getSize(1);
    size_t cols_C = C->getSize(1);

    if(is_stacked_gpu && C->is_stacked_gpu) {
        rows_A = getSize(0) * getSize(2);
        rows_C = C->getSize(0) * C->getSize(2);
        depth_A = 1;
        depth_C = 1;
        if(B->is_stacked_gpu) {
            rows_B = B->getSize(0) * B->getSize(2);
            depth_B = 1;
        }
    }
    size_t depth = max(max(depth_A, depth_B), depth_C);

    size_t rows_op_A = (op_A == CUBLAS_OP_N) ? rows_A : cols_A; 
    size_t cols_op_A = (op_A == CUBLAS_OP_N) ? cols_A : rows_A; 
    size_t rows_op_B = (op_B == CUBLAS_OP_N) ? rows_B : cols_B; 
    size_t cols_op_B = (op_B == CUBLAS_OP_N) ? cols_B : rows_B; 

    if(cols_C != cols_op_B || rows_C != rows_op_A || cols_op_A != rows_op_B
            || (depth_A > 1 && depth_A != depth)
            || (depth_B > 1 && depth_B != depth)
            || (depth_C > 1 && depth_C != depth)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    //for tall skinny op_A
    if(rows_op_A >= 8192 && cols_op_A < 256 && cols_op_B > 1 && cols_op_B < 256) {
        //for smaller dim_T_c, call my own makeshift GEMM that's somehow faster for the typical sized problem
        dim3 block_size;
        dim3 grid_size;
        if(cols_op_B > 8) { 
            block_size.y = 8;
        }
        else if(cols_op_B >= 4) { 
            block_size.y = 4;
        }
        block_size.x = 1024/block_size.y;
        grid_size.x  = getSize(0) / block_size.x + ( (getSize(0) % block_size.x == 0) ? 0 : 1);
        grid_size.y  = 1;
        block_size.z = 1;
        grid_size.z  = getSize(2);

        cudaStream_t stream;
        ce = cublasGetStream(handle, &stream);

        if(ce == CUBLAS_STATUS_SUCCESS) {
            const GPUData_kernel<FPTYPE> * aa = B->device();
            kernel_MM_quick<<<grid_size, block_size, 0, stream>>>(C->device(), device(), B->device(), alpha, beta, op_A, op_B);
            if(cudaSuccess != cudaGetLastError()) {
                ce = CUBLAS_STATUS_INVALID_VALUE;
            }
            else {
                ce = CUBLAS_STATUS_SUCCESS;
            }
        }
    }
    else if(cols_op_B == 1) {
       //GEMV is sometimes way faster then GEMM (even on the same problem size) - call it if it's all that's needed
        for(int dd = 0; dd < depth && ce == CUBLAS_STATUS_SUCCESS; dd++) {
            ce = cublasGEMV(handle, op_A,
                          rows_A, cols_A,
                          &alpha,
                          getData_gpu() + dd*inc_gpu, ld_gpu,
                          B->getData_gpu() + dd*B->inc_gpu, (op_B == CUBLAS_OP_N) ? static_cast<size_t>(1) : B->ld_gpu,
                          &beta,
                          C->getData_gpu() + dd*C->inc_gpu, 1);
        }
    }
    else if(rows_op_A < 256 && cols_op_A > 8192 && cols_op_B < 256) {
        //A'*B for tall, skinny A&B is slow with GEMM, somehow faster with multiple GEMV calls - perverse but it's a bit speedup
        size_t op_B_ld     = (op_B == CUBLAS_OP_N) ? B->ld_gpu : static_cast<size_t>(1);
        size_t op_B_stride = (op_B == CUBLAS_OP_N) ? static_cast<size_t>(1) : B->ld_gpu;
                
        for(int dd = 0; dd < depth && ce == CUBLAS_STATUS_SUCCESS; dd++) {
            for(int rr = 0; rr < C->getSize(1) && ce == CUBLAS_STATUS_SUCCESS; rr++) {
                ce = cublasGEMV(handle, op_A,
                              rows_A, cols_A,
                              &alpha,
                              getData_gpu() + dd*inc_gpu, ld_gpu,
                              B->getData_gpu() + rr*op_B_ld + dd*B->inc_gpu, op_B_stride,
                              &beta,
                              C->getData_gpu() + rr*C->ld_gpu + dd*C->inc_gpu, 1);
            }
        }
    }
    else if(depth > 1) {
        //for largish size of C, call GEMM (single - run below)
        ce = cublasGEMMStridedBatched(handle,
                                  op_A,
                                  op_B,
                                  rows_op_A, cols_op_B, cols_op_A,
                                  &alpha,
                                  getData_gpu(), ld_gpu,
                                  inc_gpu,
                                  B->getData_gpu(), B->ld_gpu,
                                  B->inc_gpu,
                                  &beta,
                                  C->getData_gpu(), C->ld_gpu,
                                  C->inc_gpu,
                                  depth);
    }
    else {
        ce = cublasGEMM(handle,
                          op_A,
                          op_B,
                          rows_op_A, cols_op_B, cols_op_A,
                          &alpha,
                          getData_gpu(), ld_gpu,
                          B->getData_gpu(), B->ld_gpu,
                          &beta,
                          C->getData_gpu(), C->ld_gpu);
    }
    return ce;
}

template <class FPTYPE>
cublasStatus_t GPUData<FPTYPE>::GEMVs(GPUData<FPTYPE> * C, const GPUData<FPTYPE> * B, const cublasHandle_t handle, const cublasOperation_t op_A, const cublasOperation_t op_B, const FPTYPE alpha, const FPTYPE beta) {
    
    cublasStatus_t ce = CUBLAS_STATUS_SUCCESS;

    size_t depth = getSize(2);
    size_t rows_A = getSize(0);
    size_t rows_B = B->getSize(0);
    size_t cols_A = getSize(1);
    size_t cols_B = B->getSize(1);

    size_t rows_op_A = (op_A == CUBLAS_OP_N) ? rows_A : cols_A; 
    size_t cols_op_A = (op_A == CUBLAS_OP_N) ? cols_A : rows_A; 
    size_t rows_op_B = (op_B == CUBLAS_OP_N) ? rows_B : cols_B; 
    size_t cols_op_B = (op_B == CUBLAS_OP_N) ? cols_B : rows_B;

    size_t op_B_ld     = (op_B == CUBLAS_OP_N) ? B->ld_gpu : static_cast<size_t>(1);
    size_t op_B_stride = (op_B == CUBLAS_OP_N) ? static_cast<size_t>(1) : B->ld_gpu;

    if(C->getSize(1) != cols_op_B || C->getSize(0) != rows_op_A || cols_op_A != rows_op_B || (depth != 1 && cols_op_B != depth)) {
        ce = CUBLAS_STATUS_INVALID_VALUE;
    }

    for(int rr = 0; rr < cols_op_B && ce == CUBLAS_STATUS_SUCCESS; rr++) {
        ce = cublasGEMV(handle, op_A,
                      rows_A, cols_A,
                      &alpha,
                      getData_gpu()    + rr*inc_gpu, ld_gpu,
                      B->getData_gpu() + rr*op_B_ld, op_B_stride,
                      &beta,
                      C->getData_gpu() + rr*C->ld_gpu, 1);
    }
    return ce;
}


//explicitly create classes for single and double precision floating point for library
template class GPUData<float>;
template class GPUData<double>;
template class GPUData<int>;
template class GPUData<char>;
template class GPUData<bool>;
template class GPUData<unsigned int>;
template class GPUData<size_t>;

template class GPUData_array<float>;
template class GPUData_array<double>;
template class GPUData_array<int>;
template class GPUData_array<char>;
template class GPUData_array<unsigned int>;
template class GPUData_array<size_t>;

template class GPUData_kernel<float>;
template class GPUData_kernel<double>;
template class GPUData_kernel<int>;
template class GPUData_kernel<char>;
template class GPUData_kernel<unsigned int>;
template class GPUData_kernel<size_t>;

template class GPUData_array_kernel<float>;
template class GPUData_array_kernel<double>;
template class GPUData_array_kernel<int>;
template class GPUData_array_kernel<char>;
template class GPUData_array_kernel<unsigned int>;
template class GPUData_array_kernel<size_t>;

};//end namespace