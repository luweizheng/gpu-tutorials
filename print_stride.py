from numba import cuda

@cuda.jit
def gpu_print(N):
    idxWithinGrid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x 
    gridStride = cuda.gridDim.x * cuda.blockDim.x
    # 从 idxWithinGrid 开始
    # 每次以整个网格线程总数为跨步数
    for i in range(idxWithinGrid, N, gridStride):
        print(i)

def main():
    gpu_print[2, 4](32)
    cuda.synchronize()

if __name__ == "__main__":
    main()