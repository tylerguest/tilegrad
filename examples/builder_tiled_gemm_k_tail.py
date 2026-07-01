from tinygrad import Tensor
from tilegrad import KernelBuilder, run

k = KernelBuilder("tilegrad_builder_tiled_gemm_k_tail", ("out", "a", "b"))
k.alloc("as", 3, "float32")
k.alloc("bs", 3, "float32")
k.alloc("acc", 1, "float32", "register")

out = k.buffer("out", shape=(2, 2))
a = k.buffer("a", shape=(2, 5))
b = k.buffer("b", shape=(5, 2))
as_tile = k.buffer("as")
bs_tile = k.buffer("bs")
acc = k.buffer("acc")

with k.range("i", 2) as i:
  with k.range("j", 2) as j:
    acc[0] = 0
    with k.range("ko", 2) as ko:
      with k.range("kk", 3) as kk:
        gk = ko * 3 + kk
        k.store(as_tile, kk, k.load_if(gk < 5, a, (i, gk)))
        k.store(bs_tile, kk, k.load_if(gk < 5, b, (gk, j)))
      k.barrier()
      with k.range("kk", 3, axis="reduce") as kk:
        acc[0] = acc[0] + as_tile[kk] * bs_tile[kk]
    out[i, j] = acc[0]

if __name__ == "__main__":
  a_t = Tensor([
    1.0, 2.0, 3.0, 4.0, 5.0,
    6.0, 7.0, 8.0, 9.0, 10.0,
  ])
  b_t = Tensor([
    11.0, 12.0,
    13.0, 14.0,
    15.0, 16.0,
    17.0, 18.0,
    19.0, 20.0,
  ])
  out_t = Tensor.empty(4)
  print(run(k, out_t, a_t, b_t).tolist())  # -> [245.0, 260.0, 620.0, 660.0]
