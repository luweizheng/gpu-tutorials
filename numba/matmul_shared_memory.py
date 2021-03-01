from numba import cuda, float32, float64
import numpy as np
import math
from time import time

# thread per block
# 每个block有 BLOCK_SIZE x BLOCK_SIZE 个元素
BLOCK_SIZE = 32

@cuda.jit
def matmul(A, B, C):
    """  矩阵乘法 C = A * B
        """
    row,col = cuda.grid(2)
    
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp

@cuda.jit
def matmul_shared_memory(A, B, C):
    """
        使用Shared Memory的矩阵乘法 C = A * B
        """
    # 在Shared Memory中定义向量
    # 向量可被整个Block的所有Thread共享
    # 必须声明向量大小和数据类型
    #print(BLOCK_SIZE)
    sA = cuda.shared.array(shape=(BLOCK_SIZE, BLOCK_SIZE), dtype=float32)
    # 必须是numba type https://numba.pydata.org/numba-doc/dev/reference/types.html#numba-types
    sB = cuda.shared.array(shape=(BLOCK_SIZE, BLOCK_SIZE), dtype=float32)
    
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    row,col = cuda.grid(2)
    tmp = 0.
    
    #     if row >= C.shape[0] and col >= C.shape[1]:
    #         # 当(x, y)越界时退出
    #         return
    #print(A[5,5])
    # 以一个 BLOCK_SIZE x BLOCK_SIZE 为单位
    for m in range(math.ceil(A.shape[1] / BLOCK_SIZE)):
        
        #         if row >= C.shape[0] or col >= C.shape[1]:
        #             sA[tx, ty] = 0
        #             sB[tx, ty] = 0
        #         else:
        if ty + m * BLOCK_SIZE>=A.shape[1]:
            sA[tx, ty] = 0
        else:
            sA[tx, ty] = A[row, ty + m * BLOCK_SIZE]
        if tx + m * BLOCK_SIZE>=B.shape[0]:
            sB[tx, ty] = 0
        else:
            sB[tx, ty] = B[tx + m * BLOCK_SIZE, col]
        cuda.syncthreads()
        
        #print(m,row,col,"sA:",sA[tx, ty],"sB:",sB[tx, ty])
        #         print()
        #         if (row==3):
        
        #             print("row:",row)
        #             print("A col:",ty + m * BLOCK_SIZE)
        #             print("col:",col)
        #             print("B row:",tx + m * BLOCK_SIZE)
        # 线程同步，等待Block中所有Thread预加载结束
        # 该函数会等待Block中所有Thread执行完之后才执行下一步
        cuda.syncthreads()
        #         if(m==0 and row==0 and col==3):
        #             for j in range(BLOCK_SIZE):
        #                 for k in range(BLOCK_SIZE):
        #                     print(j,k,"sA:",sA[j,k],"sB:",sB[j,k])
        # 此时已经将A和B的子矩阵拷贝到了sA和sB
        
        # 计算Shared Memory中的向量点积
        # 直接从Shard Memory中读取数据的延迟很低
        for n in range(BLOCK_SIZE):
            #             if(m==0 and row==0 and col==3):
            #                 print(tmp)
            tmp += sA[tx, n] * sB[n, ty]
        #             if(m==0 and row==0 and col==3):
        #                 print(m,row,col,sA[tx, n] * sB[n, ty])
        #print("tmp:",tmp)
        # 线程同步，等待Block中所有Thread计算结束
        # 这里必须同步 原因在于不同步先运行的Thread会开始下一轮的循环导致sA, sB数据出现问题
                cuda.syncthreads()
            #print(m,row,col,tmp)
#print("finaltmp:",tmp)
# 循环后得到每个BLOCK的点积之和
if row < C.shape[0] and col < C.shape[1]:
    C[row, col] = tmp


# 初始化矩阵
M = 10000
N = 8000
P = 6000
A = np.random.random((M, N)) # 随机生成的 [M x N] 矩阵
B = np.random.random((N, P))
# A = np.ones((M,N)) # 随机生成的 [M x N] 矩阵
# B = np.ones((N,P)) # 随机生成的 [N x P] 矩阵

A_device = cuda.to_device(A)
B_device = cuda.to_device(B)
C_device = cuda.device_array((M, P),dtype=np.float)#M x P] 矩阵
C_device2 = cuda.device_array((M, P),dtype=np.float)

# 执行配置
threads_per_block = (BLOCK_SIZE, BLOCK_SIZE)
blocks_per_grid_x = int(math.ceil(A.shape[0] / BLOCK_SIZE))
blocks_per_grid_y = int(math.ceil(B.shape[1] / BLOCK_SIZE))
blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

start = time()
matmul[blocks_per_grid, threads_per_block](A_device, B_device, C_device)
cuda.synchronize()
print("matmul time :" + str(time() - start))
C1 = C_device.copy_to_host()

start = time()
matmul_shared_memory[blocks_per_grid, threads_per_block](A_device, B_device, C_device)
cuda.synchronize()
print("matmul with shared memory time :" + str(time() - start))
C2 = C_device.copy_to_host()

if np.allclose(C1, C2):
    print("Get Same Result!")
cuda.current_context().reset()
