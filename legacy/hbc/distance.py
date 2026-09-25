"""Unsigned point-cloud distance and mask utilities for the hard-BC ansatz.

Mesh-free and SDF-free by design: distance is taken to the boundary POINT CLOUD,
so nothing here needs normals, watertightness, or a sign — preserving the
project's point-cloud-only property.

Distances are computed in **metres**. The dataset normalizes coordinates by a
per-axis min-max, so a distance measured in normalized space is anisotropic
(1 unit along x is a different physical length than along z) and would make a
single length scale `ell` meaningless. `phys_coords` undoes that first.
"""

import torch


def phys_coords(x_norm, coord_min, coord_scale):
    """Undo the dataset's per-axis min-max normalization: normalized -> metres."""
    return x_norm * coord_scale + coord_min


def mask(phi, ell):
    """Saturating boundary mask `m = phi / (phi + ell)`.

    0 exactly on the boundary, monotonically -> 1 in the far field, bounded in
    [0, 1). `ell` (metres) is the only new hyperparameter: it sets how far the
    constraint's influence reaches. Small `ell` keeps the interior prediction
    almost undistorted but makes the transition steep.
    """
    return phi / (phi + ell)


def subsample_cloud(cloud, max_points, generator=None):
    """Cap a cloud's size for distance queries (deterministic if `generator` given)."""
    if max_points is None or len(cloud) <= max_points:
        return cloud
    idx = torch.randperm(len(cloud), generator=generator, device=cloud.device)[:max_points]
    return cloud[idx]


# Memory budget for the pairwise difference tensor, in bytes. The chunk size is
# derived from this and the cloud size rather than fixed, so the same call is safe
# for a 2k cloud and a 60k one.
_PAIR_BUDGET = 192 * 1024 * 1024


def _auto_chunk(n_cloud, chunk=None):
    if chunk is not None:
        return max(1, chunk)
    return max(1, min(4096, _PAIR_BUDGET // max(1, n_cloud * 3 * 4)))


@torch.no_grad()
def min_dist(x, cloud, chunk=None):
    """Distance from each point of `x` (N,3) to the nearest point of `cloud` (M,3)."""
    return min_dist_and_idx(x, cloud, chunk, want_idx=False)[0]


@torch.no_grad()
def min_dist_and_idx(x, cloud, chunk=None, want_idx=True):
    """Chunked exact nearest-neighbour search. Returns (distances, indices|None).

    Deliberately NOT `torch.cdist`. cdist's default matmul path evaluates
    `d^2 = |a|^2 + |b|^2 - 2a.b`, which catastrophically cancels when a point is
    ON the cloud: with room coordinates of ~9 m, `|a|^2 ~ 81` and fp32 epsilon
    ~1e-7 give `d^2` error ~8e-6, i.e. `d` error ~5e-4 m. Measured: 0.49 mm on
    the cloud's own points. That is fatal here — it makes `m_w = phi/(phi+ell)`
    about 0.01 instead of 0 on walls, so the "exact by construction" no-slip
    would leak ~1% of the predicted velocity. The explicit difference below has
    no cancellation and returns 0 to machine precision.

    Chunked because the difference tensor is (chunk, M, 3); `_auto_chunk` sizes it
    to `_PAIR_BUDGET`. Runs under `no_grad`: the mask multiplies the network
    output, so gradients flow through that output, never through the geometry.
    """
    n = len(x)
    c = _auto_chunk(len(cloud), chunk)
    dists = torch.empty(n, device=x.device, dtype=x.dtype)
    idxs = torch.empty(n, device=x.device, dtype=torch.long) if want_idx else None

    for i in range(0, n, c):
        diff = x[i:i + c].unsqueeze(1) - cloud.unsqueeze(0)   # (c, M, 3)
        d2 = diff.pow(2).sum(dim=-1)                          # (c, M)
        if want_idx:
            best, j = d2.min(dim=1)
            idxs[i:i + c] = j
        else:
            best = d2.min(dim=1).values
        dists[i:i + c] = best.clamp_min(0).sqrt()
    return dists, idxs


@torch.no_grad()
def nearest_value(x, cloud, cloud_val, chunk=4096):
    """Value carried by the nearest cloud point — the inlet Dirichlet datum `g`.

    The nearest-index lookup is non-differentiable, which is fine: `g` is data,
    and gradients reach the network only through its own output.
    """
    _, idx = min_dist_and_idx(x, cloud, chunk, want_idx=True)
    return cloud_val[idx]
