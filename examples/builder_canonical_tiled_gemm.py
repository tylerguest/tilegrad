from tinygrad import Tensor
from tilegrad import run
from tilegrad.kernels import tiled_gemm

M = 3
N = 3
K = 5
BM = 2
BN = 2
BK = 3

k = tiled_gemm(M, N, K, BM, BN, BK)

if __name__ == "__main__":
  a_t = Tensor([
    1.0, 2.0, 3.0, 4.0, 5.0,
    6.0, 7.0, 8.0, 9.0, 10.0,
    11.0, 12.0, 13.0, 14.0, 15.0,
  ])
  b_t = Tensor([
    16.0, 17.0, 18.0,
    19.0, 20.0, 21.0, 
    22.0, 23.0, 24.0, 
    25.0, 26.0, 27.0, 
    28.0, 29.0, 30.0,
  ])
  out_t = Tensor.empty(M * N)
  print(run(k, out_t, a_t, b_t).tolist())