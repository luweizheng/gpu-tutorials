from numba import cuda
import numpy as np
import math
from time import time

@cuda.jit
def gpu_add(a, b, result, n):
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if idx < n :
        result[idx] = a[idx] + b[idx]

@cuda.jit
def gpu_add_stride(a, b, result, n):
    idxWithinGrid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x 
    gridStride = cuda.gridDim.x * cuda.blockDim.x
    # 从 idxWithinGrid 开始
    # 每次以整个网格线程总数为跨步数
    for i in range(idxWithinGrid, n, gridStride):
        result[i] = a[i] + b[i]

def main():
    n = 20000000
    x = np.arange(n).astype(np.int32)
    y = 2 * x

    x_device = cuda.to_device(x)
    y_device = cuda.to_device(y)
    out_device = cuda.device_array(n)

    threads_per_block = 1024
    blocks_per_grid = math.ceil(n / threads_per_block)
    start = time()
    gpu_add[blocks_per_grid, threads_per_block](x_device, y_device, out_device, n)
    #gpu_result = out_device.copy_to_host()
    cuda.synchronize()
    print("gpu vector add time " + str(time() - start))
    start = time()
    gpu_add_stride[80 ,threads_per_block](x_device, y_device, out_device, n)
    #gpu_result = out_device.copy_to_host()
    cuda.synchronize()
    print("gpu stride vector add time " + str(time() - start))

if __name__ == "__main__":
    main()