from tinygrad import Tensor
from tilegrad import KernelBuilder, run

k = KernelBuilder("tilegrad_builder_tiled_gemm_bm_bn_accum_tile", ("out", "a", "b"))
k.alloc("as", 6, "float32")
k.alloc("bs", 6, "float32")

out = k.buffer("out", shape=(3, 3), dtype="float32")
a = k.buffer("a", shape=(3, 5), dtype="float32")
b = k.buffer("b", shape=(5, 3), dtype="float32")
as_tile = k.buffer("as", shape=(2, 3), dtype="float32")
bs_tile = k.buffer("bs", shape=(3, 2), dtype="float32")
acc = k.fragment("acc", (2, 2), "float32")

with k.range("bi", 2) as bi:
  with k.range("bj", 2) as bj:
    k.clear(acc)
    for ko in range(2):
      k.copy(a.tile(origin=(bi*2, ko*3), shape=(2,3), bounds=(3,5)), as_tile.tile())
      k.copy(b.tile(origin=(ko*3, bj*2), shape=(3,2), bounds=(5,3)), bs_tile.tile())
      k.barrier()
      k.gemm(as_tile, bs_tile, acc)
    k.store_fragment(acc, out, (bi * 2, bj * 2), bounds=(3, 3))

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
