#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cassert>
#include <random>

constexpr int ARRAY_SIZE = 600;
constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 256;
constexpr int GRID_DIM = (ARRAY_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
constexpr int WARP_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;
__device__ __forceinline__ unsigned int get_lane_id() {
    unsigned int lane_id;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane_id));
    return lane_id;
}

__global__ void BlockScan(int* arr, int* result, int* save, int N) {
    //need to be fixed
    uint warpDim_global = (N + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ int warp_sums[WARP_PER_BLOCK];
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) {
        return;
    }
    uint warpIdx_local = threadIdx.x / WARP_SIZE;
    uint warpIdx_global = tid / WARP_SIZE;

    uint lane_id = get_lane_id();
    int value = arr[tid];
    for (int i = 1; i < WARP_SIZE; i *= 2) {
        int remote_val = __shfl_up_sync(0xffffffff, value, i); //value is the one fetching and value - i is one shuffling

        if (lane_id >= i) {
            value += remote_val;
        }
    }

    //__ffs: First 1 on least significant bit
    //__clz: Return the number of consecutive high-order zero bits in a 32-bit integer.
    unsigned int mask = __activemask();
    int leader_land_id = WARP_SIZE - 1 - __clz(mask);
    if (lane_id == leader_land_id) {
        warp_sums[warpIdx_local] = value;
    }

    __syncthreads();
    //no problem until here.
    if (warpIdx_local == 0) {//i.e. if it's on the first warp of a block.
        //each thread control a warp. 0th thread -> oth warp. 31st thread -> 31st warp
        //local_warp_id is same as lane_id
        uint warpDim_local_threshold = (blockIdx.x + 1 != GRID_DIM) ? WARP_PER_BLOCK : (warpDim_global % WARP_PER_BLOCK);
        int block_value = (lane_id < warpDim_local_threshold)? warp_sums[lane_id] : 0;
        for (int i = 1; i < warpDim_local_threshold; i *= 2) {
            int remote_val = __shfl_up_sync(0xffffffff, block_value, i);

            if (lane_id >= i) {
                block_value += remote_val;
            }
        }
        if (lane_id < warpDim_local_threshold) {
            warp_sums[lane_id] = block_value;
        }
    }
    __syncthreads();

    if (warpIdx_local > 0) {
        value += warp_sums[warpIdx_local - 1];
    }

    result[tid] = value;

    if (threadIdx.x == BLOCK_SIZE - 1 || tid == ARRAY_SIZE - 1) {
        save[blockIdx.x] = value;
    }
    __syncthreads();
}

__global__ void GlobalScan(int* save, int len) {
    __shared__ int warp_sums[WARP_PER_BLOCK];
    uint lane_id = get_lane_id();
    uint warpIdx = threadIdx.x / WARP_SIZE;
    int value = save[threadIdx.x];
    for (int i = 1; i < len; i *= 2) {
        int remote_val = __shfl_up_sync(0xffffffff, value, i);

        if (lane_id >= i) {
            value += remote_val;
        }
    }
    unsigned int mask = __activemask();
    int leader_land_id = WARP_SIZE - 1 - __clz(mask);
    if (lane_id == leader_land_id) {
        warp_sums[warpIdx] = value;
    }

    __syncthreads();

    if (warpIdx > 0) {
        value += warp_sums[warpIdx - 1];
    }

    save[threadIdx.x] = value;
}

__global__ void GlobalUpdate(int* result, int* save) {
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x > 0) {
        result[tid] += save[blockIdx.x - 1];
    }
}
int main() {
    assert(ARRAY_SIZE <= BLOCK_SIZE * BLOCK_SIZE);
    assert(BLOCK_SIZE % WARP_SIZE == 0);
    auto elements = std::vector<int>(ARRAY_SIZE);
    auto results = std::vector<int>(ARRAY_SIZE);
    auto blockLeaderSave = std::vector<int>(GRID_DIM);
    std::iota(elements.begin(), elements.end(), 0);

    std::random_device rd;
    std::mt19937 g(rd());

    std::shuffle(elements.begin(), elements.end(), g);

    std::cout << "Host list: " << std::endl;
    for (int i = 0; i < ARRAY_SIZE; i++) {
        std::cout << elements[i] << " ";
    }
    int* device_ptr;
    int* result_ptr;
    int* block_save_ptr;
    cudaMalloc(&device_ptr, ARRAY_SIZE * sizeof(int));
    cudaMalloc(&result_ptr, ARRAY_SIZE * sizeof(int));
    cudaMalloc(&block_save_ptr, GRID_DIM * sizeof(int));
    cudaMemcpy(device_ptr, elements.data(), ARRAY_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(result_ptr, 0, ARRAY_SIZE * sizeof(int));
    cudaMemset(block_save_ptr, 0, GRID_DIM * sizeof(int));

    BlockScan<<<GRID_DIM, BLOCK_SIZE>>>(device_ptr, result_ptr,block_save_ptr,ARRAY_SIZE);
    cudaDeviceSynchronize();
    GlobalScan<<<1, GRID_DIM>>>(block_save_ptr,GRID_DIM);
    cudaDeviceSynchronize();
    GlobalUpdate<<<GRID_DIM, BLOCK_SIZE>>>(result_ptr,block_save_ptr);
    cudaDeviceSynchronize();
    cudaMemcpy(results.data(), result_ptr, ARRAY_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(blockLeaderSave.data(), block_save_ptr, GRID_DIM * sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << std::endl << "Result list: " << std::endl;
    for (int i = 0; i < ARRAY_SIZE; i++) {
        std::cout << results[i] << " ";
    }
    std::cout << std::endl << "now grid dim:" << std::endl;
    cudaFree(device_ptr);
    cudaFree(result_ptr);
    cudaFree(block_save_ptr);

    return 0;
}