from tinygrad import Tensor
from tilegrad import KernelBuilder, run

k = KernelBuilder("tilegrad_builder_tiled_gemm_bm_bn_accum_tile", ("out", "a", "b"))
k.alloc("as", 6, "float32")
k.alloc("bs", 6, "float32")
k.alloc("acc", 4, "float32", "register")

out = k.buffer("out", shape=(3, 3))
a = k.buffer("a", shape=(3, 5))
b = k.buffer("b", shape=(5, 3))
as_tile = k.buffer("as", shape=(2, 3))
bs_tile = k.buffer("bs", shape=(3, 2))
acc = k.buffer("acc", shape=(2, 2))

with k.range("bi", 2) as bi:
  with k.range("bj", 2) as bj:
    with k.range("ko", 2) as ko:
      with k.range("ii", 2) as ii:
        gi = bi * 2 + ii
        with k.range("kk", 3) as kk:
          gk = ko * 3 + kk
          k.store(as_tile, (ii, kk), k.load_if((gi < 3) & (gk < 5), a, (gi, gk)))
      with k.range("kk", 3) as kk:
        gk = ko * 3 + kk
        with k.range("jj", 2) as jj:
          gj = bj * 2 + jj
          k.store(bs_tile, (kk, jj), k.load_if((gk < 5) & (gj < 3), b, (gk, gj)))
      k.barrier()
      with k.range("ii", 2) as ii:
        with k.range("jj", 2) as jj:
          k.set_if(ko < 1, acc, (ii, jj), 0)
          with k.range("kk", 3, axis="reduce") as kk:
            acc[ii, jj] = acc[ii, jj] + as_tile[ii, kk] * bs_tile[kk, jj]
    with k.range("ii", 2) as ii:
      gi = bi * 2 + ii
      with k.range("jj", 2) as jj:
        gj = bj * 2 + jj
        k.store_if((gi < 3) & (gj < 3), out, (gi, gj), acc[ii, jj])

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
