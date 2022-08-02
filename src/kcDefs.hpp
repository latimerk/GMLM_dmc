/*
 * kcDefs.hpp
 * A bunch of definitions for more convenient templated calls to different
 * CUDA/CUBLAS/CUSPARSE functions.
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
#ifndef GMLM_KCDEFS_H
#define GMLM_KCDEFS_H

#include <cmath>
#include <cuda_runtime.h>  // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>      // cusparseSpMV, cusparseSpMM
#include <cublas_v2.h>

namespace kCUDA { 

inline dim3 make_dim3(size_t x, size_t y = 1, size_t z = 1) {
    dim3 dims;
    dims.x = static_cast<int>(x);
    dims.y = static_cast<int>(y);
    dims.z = static_cast<int>(z);
    return dims;
}
    
    
inline cublasStatus_t cublasDOT(cublasHandle_t handle, int n,
                           const float           *x, int incx,
                           const float           *y, int incy,
                           float           *result) {
    return cublasSdot(handle,n,x,incx,y,incy,result);
}
inline cublasStatus_t cublasDOT(cublasHandle_t handle, int n,
                           const double           *x, int incx,
                           const double           *y, int incy,
                           double           *result) {
    return cublasDdot(handle,n,x,incx,y,incy,result);
}

inline cublasStatus_t cublasSCAL(cublasHandle_t handle, int n,
                            const float          *alpha,
                            float          *x, int incx) {
    return cublasSscal(handle, n, alpha, x, incx);
}

inline cublasStatus_t cublasSCAL(cublasHandle_t handle, int n,
                            const double          *alpha,
                            double          *x, int incx) {
    return cublasDscal(handle, n, alpha, x, incx);
}



inline cublasStatus_t cublasGEMV(cublasHandle_t handle, cublasOperation_t trans,
                           int m, int n,
                           const double          *alpha,
                           const double          *A, int lda,
                           const double          *x, int incx,
                           const double          *beta,
                           double          *y, int incy) {
    return cublasDgemv(handle,trans,m,n,alpha,A,lda,x,incx,beta,y,incy);
}
inline cublasStatus_t cublasGEMV(cublasHandle_t handle, cublasOperation_t trans,
                           int m, int n,
                           const float          *alpha,
                           const float          *A, int lda,
                           const float          *x, int incx,
                           const float          *beta,
                           float          *y, int incy) {
    return cublasSgemv(handle,trans,m,n,alpha,A,lda,x,incx,beta,y,incy);
}

               

inline cublasStatus_t cublasGEMM(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const double          *alpha,
                           const double          *A, int lda,
                           const double          *B, int ldb,
                           const double          *beta,
                           double          *C, int ldc) {
    return cublasDgemm(handle,transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
}
inline cublasStatus_t cublasGEMM(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const float          *alpha,
                           const float          *A, int lda,
                           const float          *B, int ldb,
                           const float          *beta,
                           float          *C, int ldc) {
    return cublasSgemm(handle,transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
}

inline cublasStatus_t cublasGEMMEX(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const double          *alpha,
                           const double          *A, int lda,
                           const double          *B, int ldb,
                           const double          *beta,
                           double          *C, int ldc,
                           cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT) { 
    return cublasGemmEx(handle,
                        transa, transb,
                        m,n,k,
                        alpha,
                        A, CUDA_R_64F, lda,
                        B,  CUDA_R_64F, ldb,
                        beta,
                        C, CUDA_R_64F, ldc,
                        CUBLAS_COMPUTE_64F,
                        algo);
}

//Default arguments depending on wheter tensor cores exist
#if __CUDA_ARCH__ >= 700
   const cublasGemmAlgo_t GEMM_TENSOR_OP_ALGO_32F = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
   const cublasComputeType_t GEMM_TENSOR_COMPUTE_32F =  CUBLAS_COMPUTE_32F_FAST_TF32;
#else
   const cublasGemmAlgo_t GEMM_TENSOR_OP_ALGO_32F = CUBLAS_GEMM_DEFAULT;
   const cublasComputeType_t GEMM_TENSOR_COMPUTE_32F =  CUBLAS_COMPUTE_32F;
#endif

inline cublasStatus_t cublasGEMMEX(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const float          *alpha,
                           const float          *A, int lda,
                           const float          *B, int ldb,
                           const float          *beta,
                           float          *C, int ldc,
                           cublasGemmAlgo_t algo = GEMM_TENSOR_OP_ALGO_32F) { 
    return cublasGemmEx(handle,
                        transa, transb,
                        m,n,k,
                        alpha,
                        A, CUDA_R_32F, lda,
                        B,  CUDA_R_32F, ldb,
                        beta,
                        C, CUDA_R_32F, ldc,
                        GEMM_TENSOR_COMPUTE_32F,
                        algo);
}
inline cublasStatus_t cublasGEMMEXStridedBatched(cublasHandle_t handle,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb,
                                  int m, int n, int k,
                                  const double          *alpha,
                                  const double          *A, int lda,
                                  long long int          strideA,
                                  const double          *B, int ldb,
                                  long long int          strideB,
                                  const double          *beta,
                                  double                *C, int ldc,
                                  long long int          strideC,
                                  int batchCount,
                                  cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP) {
    return cublasGemmStridedBatchedEx( handle,
                                    transa,
                                    transb,
                                    m,  n,  k,
                                    alpha,
                                    A, CUDA_R_64F, lda, strideA,
                                    B,  CUDA_R_64F, ldb, strideB,
                                    beta,
                                    C, CUDA_R_64F, ldc, strideC,
                                    batchCount,
                                    CUBLAS_COMPUTE_64F, algo);
}
inline cublasStatus_t cublasGEMMEXStridedBatched(cublasHandle_t handle,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb,
                                  int m, int n, int k,
                                  const float          *alpha,
                                  const float          *A, int lda,
                                  long long int          strideA,
                                  const float          *B, int ldb,
                                  long long int          strideB,
                                  const float          *beta,
                                  float                *C, int ldc,
                                  long long int          strideC,
                                  int batchCount,
                                  cublasGemmAlgo_t algo = GEMM_TENSOR_OP_ALGO_32F) {
    return cublasGemmStridedBatchedEx( handle,
                                    transa,
                                    transb,
                                    m,  n,  k,
                                    alpha,
                                    A, CUDA_R_32F, lda, strideA,
                                    B,  CUDA_R_32F, ldb, strideB,
                                    beta,
                                    C, CUDA_R_32F, ldc,  strideC,
                                    batchCount,
                                    GEMM_TENSOR_COMPUTE_32F, algo);
}

inline cublasStatus_t cublasDGMM(cublasHandle_t handle, cublasSideMode_t mode,  //CUBLAS_SIDE_RIGHT: A x diag(X), CUBLAS_SIDE_LEFT: diag(X) x A
                          int m, int n,
                          const double          *A, int lda,
                          const double          *x, int incx,
                          double          *C, int ldc) {
        
        return cublasDdgmm( handle,  mode,
                           m,  n,
                          A, lda,
                          x, incx,
                          C, ldc);
}
inline cublasStatus_t cublasDGMM(cublasHandle_t handle, cublasSideMode_t mode,  //CUBLAS_SIDE_RIGHT: A x diag(X), CUBLAS_SIDE_LEFT: diag(X) x A
                          int m, int n,
                          const float          *A, int lda,
                          const float          *x, int incx,
                          float          *C, int ldc) {
        
        return cublasSdgmm( handle,  mode,
                           m,  n,
                          A, lda,
                          x, incx,
                          C, ldc);
}


inline cublasStatus_t cublasSYRK(cublasHandle_t handle,
                           cublasFillMode_t uplo, cublasOperation_t trans, ////uplo = CUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_LOWER
                           int n, int k,
                           const double          *alpha,
                           const double          *A, int lda,
                           const double          *beta,
                           double          *C, int ldc) {
    return cublasDsyrk(handle, uplo, trans,  
            n,k,
            alpha, A, lda,
            beta,
            C, ldc);
}
inline cublasStatus_t cublasSYRK(cublasHandle_t handle,
                           cublasFillMode_t uplo, cublasOperation_t trans, ////uplo = CUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_LOWER
                           int n, int k,
                           const float          *alpha,
                           const float          *A, int lda,
                           const float          *beta,
                           float          *C, int ldc) {
    return cublasSsyrk(handle, uplo,  trans,
            n,k,  // n = number of rows of matrix op(A) 
            alpha, A, lda,
            beta,
            C, ldc);
}

inline cublasStatus_t cublasGEMMStridedBatched(cublasHandle_t handle,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb,
                                  int m, int n, int k,
                                  const double          *alpha,
                                  const double          *A, int lda,
                                  long long int          strideA,
                                  const double          *B, int ldb,
                                  long long int          strideB,
                                  const double          *beta,
                                  double                *C, int ldc,
                                  long long int          strideC,
                                  int batchCount) {
    return cublasDgemmStridedBatched( handle,
                                   transa,
                                   transb,
                                   m,  n,  k,
                                   alpha,
                                   A, lda,
                                   strideA,
                                   B, ldb,
                                   strideB,
                                   beta,
                                   C, ldc,
                                   strideC,
                                   batchCount);
}

 
      
 
inline cublasStatus_t cublasGEMMStridedBatched(cublasHandle_t handle,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb,
                                  int m, int n, int k,
                                  const float          *alpha,
                                  const float          *A, int lda,
                                  long long int          strideA,
                                  const float          *B, int ldb,
                                  long long int          strideB,
                                  const float          *beta,
                                  float                *C, int ldc,
                                  long long int          strideC,
                                  int batchCount) {
    return cublasSgemmStridedBatched( handle,
                                   transa,
                                   transb,
                                   m,  n,  k,
                                   alpha,
                                   A, lda,
                                   strideA,
                                   B, ldb,
                                   strideB,
                                   beta,
                                   C, ldc,
                                   strideC,
                                   batchCount);
}


inline __device__ float safeExp(float x, float d = 1.0f) {
//     return exp(min(88.7f,x));
    float max_n = 20.0f*d;
    return exp((x < max_n) ? x*d : max_n);
}
inline __device__ double safeExp(double x, double d = 1.0) {
//     return exp(min(90.0,x));
    double max_n = 90.0*d;
    return exp((x < max_n) ? x*d : max_n);
}
inline __device__ float safeExpm1(float x, float d = 1.0f) {
//     return exp(min(88.7f,x));
    float max_n = 20.0f*d;
    return expm1((x < max_n) ? x*d : max_n);
}
inline __device__ double safeExpm1(double x, double d = 1.0) {
//     return exp(min(90.0,x));
    double max_n = 90.0*d;
    return expm1((x < max_n) ? x*d : max_n);
}


// cuBLAS API errors
inline const char* cublasGetErrorString(cublasStatus_t status) {
    switch(status)
    {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE"; 
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH"; 
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED"; 
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR"; 
    }
    return "unknown error";
}


// dummy functions to make sure my rough templating can compile
inline cublasStatus_t cublasGEMM(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const char          *alpha,
                           const char          *A, int lda,
                           const char          *B, int ldb,
                           const char          *beta,
                           char          *C, int ldc) {
    return CUBLAS_STATUS_INVALID_VALUE;
}
inline cublasStatus_t cublasGEMM(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const int          *alpha,
                           const int          *A, int lda,
                           const int          *B, int ldb,
                           const int          *beta,
                           int          *C, int ldc) {
    return CUBLAS_STATUS_INVALID_VALUE;
}
inline cublasStatus_t cublasGEMM(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const unsigned int          *alpha,
                           const unsigned int          *A, int lda,
                           const unsigned int          *B, int ldb,
                           const unsigned int          *beta,
                           unsigned int          *C, int ldc) {
    return CUBLAS_STATUS_INVALID_VALUE;
}
inline cublasStatus_t cublasGEMM(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const size_t          *alpha,
                           const size_t          *A, int lda,
                           const size_t          *B, int ldb,
                           const size_t          *beta,
                           size_t          *C, int ldc) {
    return CUBLAS_STATUS_INVALID_VALUE;
}
inline cublasStatus_t cublasGEMM(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const bool          *alpha,
                           const bool          *A, int lda,
                           const bool          *B, int ldb,
                           const bool          *beta,
                           bool          *C, int ldc) {
    return CUBLAS_STATUS_INVALID_VALUE;
}


inline cublasStatus_t cublasGEMV(cublasHandle_t handle, cublasOperation_t trans,
                           int m, int n,
                           const char             *alpha,
                           const char          *A, int lda,
                           const char          *x, int incx,
                           const char          *beta,
                           char          *y, int incy) {
    return CUBLAS_STATUS_INVALID_VALUE;
}
inline cublasStatus_t cublasGEMV(cublasHandle_t handle, cublasOperation_t trans,
                           int m, int n,
                           const int             *alpha,
                           const int          *A, int lda,
                           const int          *x, int incx,
                           const int          *beta,
                           int          *y, int incy) {
    return CUBLAS_STATUS_INVALID_VALUE;
}
inline cublasStatus_t cublasGEMV(cublasHandle_t handle, cublasOperation_t trans,
                           int m, int n,
                           const unsigned int             *alpha,
                           const unsigned int          *A, int lda,
                           const unsigned int          *x, int incx,
                           const unsigned int          *beta,
                           unsigned int          *y, int incy) {
    return CUBLAS_STATUS_INVALID_VALUE;
}
inline cublasStatus_t cublasGEMV(cublasHandle_t handle, cublasOperation_t trans,
                           int m, int n,
                           const size_t             *alpha,
                           const size_t          *A, int lda,
                           const size_t          *x, int incx,
                           const size_t          *beta,
                           size_t          *y, int incy) {
    return CUBLAS_STATUS_INVALID_VALUE;
}
inline cublasStatus_t cublasGEMV(cublasHandle_t handle, cublasOperation_t trans,
                           int m, int n,
                           const bool             *alpha,
                           const bool          *A, int lda,
                           const bool          *x, int incx,
                           const bool          *beta,
                           bool          *y, int incy) {
    return CUBLAS_STATUS_INVALID_VALUE;
}

inline cublasStatus_t cublasGEMMStridedBatched(cublasHandle_t handle,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb,
                                  int m, int n, int k,
                                  const char          *alpha,
                                  const char          *A, int lda,
                                  long long int          strideA,
                                  const char          *B, int ldb,
                                  long long int          strideB,
                                  const char          *beta,
                                  char                *C, int ldc,
                                  long long int          strideC,
                                  int batchCount) {
    return CUBLAS_STATUS_INVALID_VALUE;
}
inline cublasStatus_t cublasGEMMStridedBatched(cublasHandle_t handle,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb,
                                  int m, int n, int k,
                                  const size_t          *alpha,
                                  const size_t          *A, int lda,
                                  long long int          strideA,
                                  const size_t          *B, int ldb,
                                  long long int          strideB,
                                  const size_t          *beta,
                                  size_t                *C, int ldc,
                                  long long int          strideC,
                                  int batchCount) {
    return CUBLAS_STATUS_INVALID_VALUE;
}
inline cublasStatus_t cublasGEMMStridedBatched(cublasHandle_t handle,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb,
                                  int m, int n, int k,
                                  const bool          *alpha,
                                  const bool          *A, int lda,
                                  long long int          strideA,
                                  const bool          *B, int ldb,
                                  long long int          strideB,
                                  const bool          *beta,
                                  bool                *C, int ldc,
                                  long long int          strideC,
                                  int batchCount) {
    return CUBLAS_STATUS_INVALID_VALUE;
}
inline cublasStatus_t cublasGEMMStridedBatched(cublasHandle_t handle,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb,
                                  int m, int n, int k,
                                  const unsigned int          *alpha,
                                  const unsigned int          *A, int lda,
                                  long long int          strideA,
                                  const unsigned int          *B, int ldb,
                                  long long int          strideB,
                                  const unsigned int          *beta,
                                  unsigned int                *C, int ldc,
                                  long long int          strideC,
                                  int batchCount) {
    return CUBLAS_STATUS_INVALID_VALUE;
}
inline cublasStatus_t cublasGEMMStridedBatched(cublasHandle_t handle,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb,
                                  int m, int n, int k,
                                  const int          *alpha,
                                  const int          *A, int lda,
                                  long long int          strideA,
                                  const int          *B, int ldb,
                                  long long int          strideB,
                                  const int          *beta,
                                  int                *C, int ldc,
                                  long long int          strideC,
                                  int batchCount) {
    return CUBLAS_STATUS_INVALID_VALUE;
}


inline cublasStatus_t cublasGEMMEXStridedBatched(cublasHandle_t handle,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb,
                                  int m, int n, int k,
                                  const int          *alpha,
                                  const int          *A, int lda,
                                  long long int          strideA,
                                  const int          *B, int ldb,
                                  long long int          strideB,
                                  const int          *beta,
                                  int                *C, int ldc,
                                  long long int          strideC,
                                  int batchCount,
                                  cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP) {
    return CUBLAS_STATUS_INVALID_VALUE;
}
inline cublasStatus_t cublasGEMMEXStridedBatched(cublasHandle_t handle,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb,
                                  int m, int n, int k,
                                  const unsigned int          *alpha,
                                  const unsigned int          *A, int lda,
                                  long long int          strideA,
                                  const unsigned int          *B, int ldb,
                                  long long int          strideB,
                                  const unsigned int          *beta,
                                  unsigned int                *C, int ldc,
                                  long long int          strideC,
                                  int batchCount,
                                  cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP) {
    return CUBLAS_STATUS_INVALID_VALUE;
}
inline cublasStatus_t cublasGEMMEXStridedBatched(cublasHandle_t handle,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb,
                                  int m, int n, int k,
                                  const bool          *alpha,
                                  const bool          *A, int lda,
                                  long long int          strideA,
                                  const bool          *B, int ldb,
                                  long long int          strideB,
                                  const bool          *beta,
                                  bool                *C, int ldc,
                                  long long int          strideC,
                                  int batchCount,
                                  cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP) {
    return CUBLAS_STATUS_INVALID_VALUE;
}
inline cublasStatus_t cublasGEMMEXStridedBatched(cublasHandle_t handle,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb,
                                  int m, int n, int k,
                                  const char          *alpha,
                                  const char          *A, int lda,
                                  long long int          strideA,
                                  const char          *B, int ldb,
                                  long long int          strideB,
                                  const char          *beta,
                                  char                *C, int ldc,
                                  long long int          strideC,
                                  int batchCount,
                                  cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP) {
    return CUBLAS_STATUS_INVALID_VALUE;
}
inline cublasStatus_t cublasGEMMEXStridedBatched(cublasHandle_t handle,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb,
                                  int m, int n, int k,
                                  const size_t          *alpha,
                                  const size_t          *A, int lda,
                                  long long int          strideA,
                                  const size_t          *B, int ldb,
                                  long long int          strideB,
                                  const size_t          *beta,
                                  size_t                *C, int ldc,
                                  long long int          strideC,
                                  int batchCount,
                                  cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP) {
    return CUBLAS_STATUS_INVALID_VALUE;
}
inline cublasStatus_t cublasGEMMEX(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const size_t          *alpha,
                           const size_t          *A, int lda,
                           const size_t          *B, int ldb,
                           const size_t          *beta,
                           size_t          *C, int ldc,
                           cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP) { 
    return CUBLAS_STATUS_INVALID_VALUE;
}
inline cublasStatus_t cublasGEMMEX(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const char          *alpha,
                           const char          *A, int lda,
                           const char          *B, int ldb,
                           const char          *beta,
                           char          *C, int ldc,
                           cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP) { 
    return CUBLAS_STATUS_INVALID_VALUE;
}
inline cublasStatus_t cublasGEMMEX(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const unsigned int          *alpha,
                           const unsigned int           *A, int lda,
                           const unsigned int           *B, int ldb,
                           const unsigned int           *beta,
                           unsigned int           *C, int ldc,
                           cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP) { 
    return CUBLAS_STATUS_INVALID_VALUE;
}
inline cublasStatus_t cublasGEMMEX(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const bool          *alpha,
                           const bool          *A, int lda,
                           const bool          *B, int ldb,
                           const bool          *beta,
                           bool          *C, int ldc,
                           cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP) { 
    return CUBLAS_STATUS_INVALID_VALUE;
}
inline cublasStatus_t cublasGEMMEX(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const int          *alpha,
                           const int          *A, int lda,
                           const int          *B, int ldb,
                           const int          *beta,
                           int          *C, int ldc,
                           cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP) { 
    return CUBLAS_STATUS_INVALID_VALUE;
}
};//end samespace

#endif
