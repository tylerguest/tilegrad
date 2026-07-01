from tinygrad import Tensor
from tilegrad import KernelBuilder, run

k = KernelBuilder("tilegrad_builder_tiled_gemm_full_edge_tile", ("out", "a", "b"))
k.alloc("as", 3, "float32")
k.alloc("bs", 3, "float32")
k.alloc("acc", 1, "float32", "register")

out = k.buffer("out", shape=(3, 3))
a = k.buffer("a", shape=(3, 5))
b = k.buffer("b", shape=(5, 3))
as_tile = k.buffer("as")
bs_tile = k.buffer("bs")
acc = k.buffer("acc")

with k.range("i", 4) as i:
  with k.range("j", 4) as j:
    acc[0] = 0
    with k.range("ko", 2) as ko:
      with k.range("kk", 3) as kk:
        gk = ko * 3 + kk
        k.store(as_tile, kk, k.load_if((i < 3) & (gk < 5), a, (i, gk)))
        k.store(bs_tile, kk, k.load_if((gk < 5) & (j < 3), b, (gk, j)))
      k.barrier()
      with k.range("kk", 3, axis="reduce") as kk:
        acc[0] = acc[0] + as_tile[kk] * bs_tile[kk]
    k.store_if((i < 3) & (j < 3), out, (i, j), acc[0])

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
  out_t = Tensor.empty(9)
  print(run(k, out_t, a_t, b_t).tolist())
