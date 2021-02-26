import numpy as np
import math
from time import time
from numba import cuda
from numba import jit
import matplotlib
# 使用无显示器的服务器进行计算时，需加上下面这行，否则matplot报错
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def timeit(func ,number_of_iterations, input_args):
    # 计时函数
    start = time()
    for i in range(number_of_iterations):
        func(*input_args)
    stop = time()

    return stop - start

def randfloat(rand_var, low, high):
    # 随机函数
    return (1.0 - rand_var) * low + rand_var * high

RISKFREE = 0.02
VOLATILITY = 0.30

def cnd(d):
    # 正态分布累计概率分布函数
    A1 = 0.31938153
    A2 = -0.356563782
    A3 = 1.781477937
    A4 = -1.821255978
    A5 = 1.330274429
    RSQRT2PI = 0.39894228040143267793994605993438
    K = 1.0 / (1.0 + 0.2316419 * np.abs(d))
    ret_val = (RSQRT2PI * np.exp(-0.5 * d * d) *
               (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5))))))
    return np.where(d > 0, 1.0 - ret_val, ret_val)

def black_scholes(stockPrice, optionStrike, optionYears, riskFree, volatility):
    # Python + Numpy 实现B-S模型
    S = stockPrice
    K = optionStrike
    T = optionYears
    r = riskFree
    V = volatility
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * V * V) * T) / (V * sqrtT)
    d2 = d1 - V * sqrtT
    cndd1 = cnd(d1)
    cndd2 = cnd(d2)

    expRT = np.exp(- r * T)

    callResult = S * cndd1 - K * expRT * cndd2
    putResult = K * expRT * (1.0 - cndd2) - S * (1.0 - cndd1)

    return callResult, putResult

@cuda.jit(device=True)
def cnd_cuda(d):
    # 正态分布累计概率分布函数
    # CUDA设备端函数
    A1 = 0.31938153
    A2 = -0.356563782
    A3 = 1.781477937
    A4 = -1.821255978
    A5 = 1.330274429
    RSQRT2PI = 0.39894228040143267793994605993438
    K = 1.0 / (1.0 + 0.2316419 * math.fabs(d))
    ret_val = (RSQRT2PI * math.exp(-0.5 * d * d) *
               (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5))))))
    if d > 0:
        ret_val = 1.0 - ret_val
    return ret_val


@cuda.jit
def black_scholes_cuda_kernel(callResult, putResult, S, K,
                       T, r, V):
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if i >= S.shape[0]:
        return
    sqrtT = math.sqrt(T[i])
    d1 = (math.log(S[i] / K[i]) + (r + 0.5 * V * V) * T[i]) / (V * sqrtT)
    d2 = d1 - V * sqrtT
    cndd1 = cnd_cuda(d1)
    cndd2 = cnd_cuda(d2)

    expRT = math.exp((-1. * r) * T[i])
    callResult[i] = (S[i] * cndd1 - K[i] * expRT * cndd2)
    putResult[i] = (K[i] * expRT * (1.0 - cndd2) - S[i] * (1.0 - cndd1))

def black_scholes_cuda(stockPrice, optionStrike,
                        optionYears, riskFree, volatility):
    # CUDA实现B-S模型
    blockdim = 1024
    griddim = int(math.ceil(float(len(stockPrice))/blockdim))

    device_callResult = cuda.device_array_like(stockPrice)
    device_putResult = cuda.device_array_like(stockPrice)
    device_stockPrice = cuda.to_device(stockPrice)
    device_optionStrike = cuda.to_device(optionStrike)
    device_optionYears = cuda.to_device(optionYears)

    black_scholes_cuda_kernel[griddim, blockdim](
            device_callResult, device_putResult, device_stockPrice, device_optionStrike,
            device_optionYears, riskFree, volatility)
    callResult = device_callResult.copy_to_host()
    putResult= device_putResult.copy_to_host()
    cuda.synchronize()

    return callResult, putResult

def main():
    pure_time_list = []
    cuda_time_list = []

    dtype = np.float32
    data_size = [10, 1000, 100000, 1000000, 4000000]
    for OPT_N in data_size:
        print("data size :" + str(OPT_N))

        stockPrice = randfloat(np.random.random(OPT_N), 5.0, 30.0).astype(dtype)
        optionStrike = randfloat(np.random.random(OPT_N), 1.0, 100.0).astype(dtype)
        optionYears = randfloat(np.random.random(OPT_N), 0.25, 10.0).astype(dtype)

        input_args=(stockPrice, optionStrike, optionYears, RISKFREE, VOLATILITY)
        pure_duration = timeit(black_scholes, 20, input_args)

        pure_time_list.append(pure_duration)
        cuda_duration = timeit(black_scholes_cuda, 20, input_args)
        cuda_time_list.append(cuda_duration)

    print(pure_time_list)
    print(cuda_time_list)
    plt.plot(pure_time_list[1:], label='pure python')
    plt.plot(cuda_time_list[1:], label='cuda')
    plt.legend(loc='upper right')
    plt.xticks(np.arange(5), ('1000', '100000', '1000000', '4000000'))
    #设置坐标轴名称
    plt.ylabel('duration (second)')
    plt.xlabel('option number')
    plt.savefig("benchmark.png")
    plt.show()

if __name__ == "__main__":
    main()