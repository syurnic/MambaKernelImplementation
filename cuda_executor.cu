#include <cuda_fp16.h>
#include <iostream>
#include <vector>

constexpr int BlockSize = 256;

__global__ void MambaScanKernel(
    /*
     * Idea: On Grid level, data is split by Batch and Channel.
     * On Block level(Each block is responsible for one total sequence in 1 batch 1 channel):
     * Each block chunks long length of L into 256/512 blocks.
     * Set 1 block as 256 threads.
     */
    const __half* __restrict__ A, //(D,N)
    const __half* __restrict__ B_mat, //(B,L,D)
    const __half* __restrict__ C_mat, //(B,L,D)
    const __half* __restrict__ u, //(B,L,D)
    const __half* __restrict__ delta, //(B,L,D)
    __half* __restrict__ y, //(B,L,D)
    const int B,
    const int L,
    const int N,
    const int D
    //Should check shape before kernel execution
    //calculation of B,C,delta should be done before kernel execution
    ) {
    uint id = threadIdx.x + blockIdx.x * blockDim.x;
    uint tid = threadIdx.x;
    uint bid = blockIdx.x;
    uint this_B = blockIdx.x / D;
    uint this_D = blockIdx.x % D;
    uint this_L = threadIdx.x;
    //A_bar, B_bar should be (B, L, D, N)

    //first delta (B,L,D) and A (D,N) to deltaA(B,L,D,N)
    for (uint i = this_L; i < L; i += BlockSize) {

    }
}