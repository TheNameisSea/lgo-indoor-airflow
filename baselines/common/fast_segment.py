"""Vectorized replacement for neuralop's `segment_csr` fallback.

WHY THIS EXISTS
---------------
`neuralop.layers.segment_csr.segment_csr` uses `torch_scatter` when available and
otherwise falls back to:

    for i in range(n_out):          # n_out = 13824 grid points, or 5120 queries
        out[i] += einsum("io->o", src[indptr[i]:indptr[i+1]])

i.e. one Python iteration and one CUDA kernel launch PER OUTPUT POINT, every layer,
every forward and backward. Measured effect on this task: 6.6 s/case for GINO and
9.4 s/case for GNO at only ~1.5 GiB peak memory — launch-bound, not memory-bound,
which is what made a 500-epoch sweep run cost 11-16 h.

`torch_scatter` is the upstream fix but is documented as conflicting with recent
PyTorch (we are on 2.13.0+cu132), so instead this module patches in an equivalent
`index_add_` implementation: one kernel for the whole reduction, differentiable,
and numerically identical up to float addition order.

`verify()` checks it against the original implementation; `patch()` installs it.
Both `neuralop.layers.segment_csr` and the already-bound reference inside
`neuralop.layers.integral_transform` are patched, since the latter did
`from .segment_csr import segment_csr` at import time.
"""

import torch

_original = None


def segment_csr_fast(src, indptr, reduction="sum", use_scatter=None):
    """CSR segment reduction. src: (P, C) or (B, P, C); indptr: (n_out+1,) or (B, n_out+1)."""
    if reduction not in ("mean", "sum"):
        raise ValueError("reduce must be one of 'mean', 'sum'")

    if indptr.ndim == 2:
        indptr = indptr[0]
    n_out = indptr.shape[0] - 1
    counts = indptr[1:] - indptr[:-1]
    row = torch.repeat_interleave(
        torch.arange(n_out, device=src.device), counts)

    if src.ndim == 3:
        out = src.new_zeros(src.shape[0], n_out, src.shape[-1])
        out = out.index_add(1, row, src)
        if reduction == "mean":
            out = out / counts.clamp(min=1).to(src.dtype).view(1, -1, 1)
    else:
        out = src.new_zeros(n_out, src.shape[-1])
        out = out.index_add(0, row, src)
        if reduction == "mean":
            out = out / counts.clamp(min=1).to(src.dtype).view(-1, 1)
    return out


def patch():
    """Install the fast implementation into neuralop (idempotent)."""
    global _original
    import neuralop.layers.integral_transform as it
    import neuralop.layers.segment_csr as sc

    if _original is None:
        _original = sc.segment_csr
    sc.segment_csr = segment_csr_fast
    it.segment_csr = segment_csr_fast


def verify(device="cuda", tol=2e-4):
    """Check the fast path against neuralop's original loop implementation."""
    import neuralop.layers.segment_csr as sc
    orig = _original or sc.segment_csr

    torch.manual_seed(0)
    failures = []
    for n_out, batched in ((512, False), (512, True), (37, False)):
        counts = torch.randint(0, 9, (n_out,), device=device)
        indptr = torch.cat([torch.zeros(1, dtype=torch.long, device=device),
                            counts.cumsum(0)])
        total = int(indptr[-1].item())
        shape = (2, total, 7) if batched else (total, 7)
        src = torch.randn(*shape, device=device)
        ind = indptr.unsqueeze(0).repeat(2, 1) if batched else indptr

        for red in ("sum", "mean"):
            a = orig(src, ind, reduction=red, use_scatter=False)
            b = segment_csr_fast(src, ind, reduction=red)
            if a.shape != b.shape:
                failures.append(f"{red} n_out={n_out} batched={batched}: "
                                f"shape {tuple(a.shape)} vs {tuple(b.shape)}")
            else:
                d = (a - b).abs().max().item()
                if d > tol:
                    failures.append(f"{red} n_out={n_out} batched={batched}: "
                                    f"max|diff|={d:.2e}")

    # Gradients must match too.
    counts = torch.randint(1, 9, (128,), device=device)
    indptr = torch.cat([torch.zeros(1, dtype=torch.long, device=device),
                        counts.cumsum(0)])
    src = torch.randn(int(indptr[-1]), 5, device=device)
    g = []
    for fn in (orig, segment_csr_fast):
        s = src.clone().requires_grad_(True)
        fn(s, indptr, reduction="mean", use_scatter=False).square().sum().backward()
        g.append(s.grad)
    d = (g[0] - g[1]).abs().max().item()
    if d > tol:
        failures.append(f"grad: max|diff|={d:.2e}")

    return failures


if __name__ == "__main__":
    import neuralop.layers.segment_csr as sc
    _original = sc.segment_csr
    fails = verify()
    print("segment_csr_fast:", "PASS" if not fails else "FAIL")
    for f in fails:
        print("  ", f)
