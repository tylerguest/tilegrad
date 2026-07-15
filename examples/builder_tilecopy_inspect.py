from tinygrad import Tensor
from tilegrad import KernelBuilder, run
from tilegrad.tiles import expand_tile_copies
from tilegrad.ir import TileCopy

k = KernelBuilder("tilecopy_inspect", ("out", "inp"))
out = k.buffer("out", shape=(2,3), dtype="float32")
inp = k.buffer("inp", shape=(4,5), dtype="float32")

k.copy(
  inp.tile(origin=(1,2), shape=(2,3), bounds=(4,5)),
  out.tile(),
)

ir = k.build()
expanded_ir = expand_tile_copies(ir)

print("Tile IR:")
print(ir)
print()

print("First op is TileCopy:")
print(isinstance(ir.body[0], TileCopy))
print()

print("Expanded scalar fallback IR:")
print(expanded_ir)
print()

if __name__ == "__main__":
  inp_t = Tensor([
    1.0, 2.0, 3.0, 4.0, 5.0,
    6.0, 7.0, 8.0, 9.0, 10.0,
    11.0, 12.0, 13.0, 14.0, 15.0,
    16.0, 17.0, 18.0, 19.0, 20.0,
  ])
  out_t = Tensor.empty(6)
  out = run(k, out_t, inp_t)
  print("Result:")
  print(out.tolist())