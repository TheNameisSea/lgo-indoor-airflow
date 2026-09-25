"""HardBCModel — distance-function ansatz that satisfies wall/inlet BCs exactly.

Wraps a built PI-GINOT `Trunk` and rewrites its output so that no-slip walls
(stage 1) and inlet velocities (stage 2) hold **by construction**, replacing the
soft `lambda_wall` / `lambda_inlet` penalties.

Math (all in physical units; see PLAN_HARD_BC_UPGRADE.md §2)::

    u_phys = sigma * N(x) + mu                       # de-normalized network output
    m_w    = phi_w / (phi_w + ell_w)                 # 0 on walls  -> 1 far field
    m_i    = phi_i / (phi_i + ell_i)                 # 0 on inlets -> 1 far field

    stage 1:  u_hat = m_w * u_phys
    stage 2:  u_hat = (1 - m_i) * u_in_nearest + m_w * m_i * u_phys

and the wrapper returns `(u_hat - mu) / sigma`, i.e. normalized units, so every
existing loss and metric works unchanged.

The ansatz is applied in PHYSICAL units on purpose: targets are z-scored, so a
physical `u = 0` corresponds to normalized `-mu/sigma`, not 0. Masking normalized
outputs would enforce the wrong condition.

TWO DEVIATIONS FROM THE PLAN'S SKETCH, both forced by the existing code:

1. The plan wraps `forward(xyt, pc)`. The real trainer never calls it — it calls
   `encode_geometry(pc)` once per case and then `decode_query(latent, xyt)`
   (pi_ginot/trainer.py:269,323), and `decode_query` receives no `pc`. So the
   wrapper implements all three: `encode_geometry` resolves and remembers the
   case, `decode_query` applies the ansatz. This keeps it a true drop-in for the
   existing trainer AND for `baselines/common/eval.py`.

2. The plan's `set_case(...)` reads wall/inlet pools off the batch, but
   `__getitem__` does not return them (only `pc`, `xyt`, `targets`, `bc_mask`,
   ...). Instead `register_cases(dataset)` indexes the dataset's room dicts by a
   fingerprint of `pc_full`, so the FULL boundary pools are used and **no trainer
   subclass is required at all**. If a case was never registered, the wrapper
   falls back to deriving the clouds from `pc`'s one-hot channels (the plan's
   "plan B") — which needs `pc_mode="full"`.
"""

import torch
import torch.nn as nn

from .distance import (mask, min_dist, min_dist_and_idx, nearest_value,
                       phys_coords, subsample_cloud)

# pc_full channel layout (pi_ginot/dataset.py): xyz | class one-hot(5) | u,v,w,p
_OH = slice(3, 8)
_UVWP = slice(8, 12)
_CLS_LEAK, _CLS_WALL, _CLS_INLET, _CLS_OUTLET = 0, 1, 2, 3


class HardBCModel(nn.Module):
    def __init__(self, trunk, target_mean, target_std, coord_min, coord_scale,
                 ell_wall=0.05, ell_inlet=0.05, stage=1, cloud_max=None,
                 chunk=None, seed=0):
        super().__init__()
        if stage not in (0, 1, 2):
            raise ValueError(f"stage must be 0, 1 or 2, got {stage}")
        self.trunk = trunk
        self.stage = stage
        self.ell_wall = float(ell_wall)
        self.ell_inlet = float(ell_inlet)
        # cloud_max=None (no subsample) is the default deliberately. No-slip is
        # exact only at points contained IN the distance cloud: the trainer draws
        # wall query points from `pool_wall_xyz` (up to 60k), so capping the cloud
        # at 20k leaves ~2/3 of them off-cloud with phi of a few mm, giving
        # m_w = phi/(phi+ell) ~ 0.09 and a MEASURED 0.43 m/s no-slip violation on
        # real wall points. The plan's "subsampling error << ell" holds for
        # interior points (4.5 mm) but not at the walls themselves, which is
        # precisely where exactness is the claim.
        self.cloud_max = cloud_max
        self.chunk = chunk
        self._gen = torch.Generator().manual_seed(seed)

        # Global (not per-case) normalization constants, from the shared stats.
        self.register_buffer("mu", target_mean[0:3].clone().float())
        self.register_buffer("sigma", target_std[0:3].clone().float())
        self.register_buffer("coord_min", coord_min.clone().float())
        self.register_buffer("coord_scale", coord_scale.clone().float())

        self._cases = {}        # fingerprint -> dict(wall, inlet, inlet_vel_phys)
        self._active = None     # set by encode_geometry, consumed by decode_query

    # ------------------------------------------------------------------ #
    #  Case registration
    # ------------------------------------------------------------------ #
    @staticmethod
    def _fingerprint(pc_xyz):
        """Order-invariant integer id of a boundary cloud, stable within a process."""
        q = (pc_xyz.detach().cpu().double() * 1e6).round().to(torch.int64)
        return (int(q.shape[0]), int(q.sum().item()), int((q % 9973).sum().item()))

    def register_cases(self, dataset):
        """Index a dataset's FULL boundary pools by `pc_full` fingerprint.

        Uses the *processed* room dicts, so reflection-augmented cases register
        their own mirrored clouds automatically.
        """
        n_new = 0
        for room in dataset.cases:
            key = self._fingerprint(room["pc_full"][:, :3])
            if key in self._cases:
                continue
            # Wall bucket = everything the loader treats as no-slip. Leak and
            # outlet points are excluded: they carry nonzero flow.
            wall = room["pool_wall_xyz"]
            inlet = room["pool_in_xyz"]
            inlet_tgt = room["pool_in_tgt"]
            self._cases[key] = {
                "wall_norm": wall.clone(),
                "inlet_norm": inlet.clone(),
                "inlet_tgt": inlet_tgt.clone(),
                "source": "pools",
            }
            n_new += 1
        return n_new

    def _clouds_from_pc(self, pc):
        """Fallback: derive clouds from pc's one-hot channels (plan B)."""
        if pc.shape[-1] < 12:
            raise ValueError(
                "HardBCModel fallback needs pc_mode='full' (12 channels); got "
                f"{pc.shape[-1]}. Call register_cases(dataset) instead.")
        oh = pc[:, _OH]
        cls = oh.argmax(dim=-1)
        return {
            "wall_norm": pc[cls == _CLS_WALL, :3].clone(),
            "inlet_norm": pc[cls == _CLS_INLET, :3].clone(),
            "inlet_tgt": pc[cls == _CLS_INLET, _UVWP].clone(),
            "source": "pc",
        }

    def _resolve(self, pc):
        """Look up (or derive) this case's clouds and cache them on the device."""
        pc0 = pc[0] if pc.dim() == 3 else pc
        key = self._fingerprint(pc0[:, :3])
        entry = self._cases.get(key)
        if entry is None:
            entry = self._clouds_from_pc(pc0)
            self._cases[key] = entry

        dev = pc.device
        if entry.get("_dev") != dev:
            wall = subsample_cloud(entry["wall_norm"].to(dev), self.cloud_max, None)
            inlet = entry["inlet_norm"].to(dev)
            entry["wall_phys"] = phys_coords(wall, self.coord_min, self.coord_scale)
            entry["inlet_phys"] = phys_coords(inlet, self.coord_min, self.coord_scale)
            # Inlet Dirichlet values in PHYSICAL m/s.
            tgt = entry["inlet_tgt"].to(dev)[:, 0:3]
            entry["inlet_vel_phys"] = tgt * self.sigma + self.mu
            entry["_dev"] = dev
        return entry

    # ------------------------------------------------------------------ #
    #  Drop-in Trunk interface
    # ------------------------------------------------------------------ #
    def encode_geometry(self, pc, sample_ids=None):
        self._active = None if self.stage == 0 else self._resolve(pc)
        return self.trunk.encode_geometry(pc, sample_ids=sample_ids)

    def decode_query(self, latent, xyt):
        n = self.trunk.decode_query(latent, xyt)
        if self.stage == 0:
            return n
        if self._active is None:
            raise RuntimeError("decode_query called before encode_geometry; the "
                               "wrapper needs the case set by encode_geometry.")
        return self._apply_ansatz(n, xyt, self._active)

    def forward(self, xyt, pc, sample_ids=None):
        latent = self.encode_geometry(pc, sample_ids=sample_ids)
        return self.decode_query(latent, xyt)

    # ------------------------------------------------------------------ #
    #  The ansatz
    # ------------------------------------------------------------------ #
    def _apply_ansatz(self, n, xyt, case):
        """Rewrite normalized network output `n` to satisfy the BCs exactly."""
        squeeze = (n.dim() == 3)
        n_flat = n.reshape(-1, n.shape[-1])
        x_flat = xyt.reshape(-1, xyt.shape[-1])

        x_phys = phys_coords(x_flat, self.coord_min, self.coord_scale)
        u_phys = n_flat[:, 0:3] * self.sigma + self.mu

        phi_w = min_dist(x_phys, case["wall_phys"], self.chunk)
        m_w = mask(phi_w, self.ell_wall).unsqueeze(-1)

        if self.stage == 2 and len(case["inlet_phys"]) > 0:
            # One search, not two: the distance and the nearest-point index come
            # from the same pass (calling min_dist then nearest_value would repeat
            # the whole neighbour search).
            phi_i, idx_i = min_dist_and_idx(x_phys, case["inlet_phys"], self.chunk)
            m_i = mask(phi_i, self.ell_inlet).unsqueeze(-1)
            g = (1.0 - m_i) * case["inlet_vel_phys"][idx_i]
            u_hat = g + m_w * m_i * u_phys
        else:
            u_hat = m_w * u_phys

        out = n_flat.clone()
        out[:, 0:3] = (u_hat - self.mu) / self.sigma
        return out.reshape(n.shape) if squeeze else out

    # ------------------------------------------------------------------ #
    #  Diagnostics
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def bc_residual(self, pc, xyt_norm, kind="wall"):
        """|u_hat| on wall points, or |u_hat - u_in| on inlet points, in m/s.

        This is the "by construction" receipt reported by eval_hbc.py.
        """
        case = self._resolve(pc)
        latent = self.trunk.encode_geometry(pc)
        n = self.trunk.decode_query(latent, xyt_norm)
        out = self._apply_ansatz(n, xyt_norm, case) if self.stage else n

        flat = out.reshape(-1, out.shape[-1])
        u_phys = flat[:, 0:3] * self.sigma + self.mu
        if kind == "wall":
            return u_phys.norm(dim=-1)

        x_phys = phys_coords(xyt_norm.reshape(-1, 3), self.coord_min, self.coord_scale)
        target = nearest_value(x_phys, case["inlet_phys"],
                               case["inlet_vel_phys"], self.chunk)
        return (u_phys - target).norm(dim=-1)

    def describe(self):
        return (f"HardBCModel(stage={self.stage}, ell_wall={self.ell_wall} m"
                + (f", ell_inlet={self.ell_inlet} m" if self.stage == 2 else "")
                + f", cloud_max={self.cloud_max}, cases={len(self._cases)})")
