import torch

import triton
import triton.language as tl

@triton.jit
def eltadd(x1, x2, output, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size
    x1_ptrs = x1 + offsets
    x2_ptrs = x2 + offsets
    output_ptrs = output + offsets
    _x1 = tl.load(x1_ptrs, mask=mask)
    _x2 = tl.load(x2_ptrs, mask=mask)
    _output = _x1 + _x2
    tl.store(output_ptrs, _output)

x = torch.randint(0, 1024, (1024,), dtype=torch.int32, device='cuda')
y = torch.randint(0, 1024, (1024,), dtype=torch.int32, device='cuda')
output = torch.empty_like(x, device='cuda')
out_ref = x+y
grid = lambda meta: (triton.cdiv(x.shape[0], meta['BLOCK_SIZE']),)
eltadd[grid](x, y, output, x.shape[0], BLOCK_SIZE=1024)