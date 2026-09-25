"""EquiJetModel — jet template + GINOT correction, optionally composed with hard BCs.

Implements PLAN_EQUIJET_UPGRADE.md §3.3 and the composed spec of §6.

    u_phys = m_wall(x) * ( sum_k J(x - x_k, c_k) + sigma * N(x, pc) + mu )
    return  (u_phys - mu) / sigma

With `hbc=None` the mask is identity and this is EquiJet-only (arm 3 of the §6.4
matrix); with an `hbc` provider it is EquiJet+HBC (arm 4).

DESIGN NOTES

* **encode_geometry / decode_query, not just forward.** The plan sketches a
  `forward(xyt, pc)` drop-in, but `HybridGINOTTrainer` never calls it — it calls
  `encode_geometry(pc)` once per case then `decode_query(latent, xyt)`
  (`pi_ginot/trainer.py:269,323`), and `decode_query` gets no `pc`. So the case is
  resolved and cached in `encode_geometry`. Same correction as the HBC wrapper.

* **`register_cases(dataset)` instead of a `set_case` hook.** The batch dict does
  not carry `pool_in_xyz`/`pool_in_tgt`, so vents are indexed by a fingerprint of
  `pc_full` and looked up automatically. No trainer changes are needed, and
  reflection-augmented rooms register their own mirrored anchors and flipped
  direction vectors because everything is read from the *processed* room dict.

* **One physical-space path (§6.2).** Composition de-normalizes once, adds the
  jets, applies the wall mask to the SUM (the jet must also vanish on the ceiling
  it hugs), and re-normalizes once — no double round-trip through the z-scores.
"""

import torch
import torch.nn as nn

from .vents import COND_DIM, extract_vents, normalize_cond


class EquiJetModel(nn.Module):
    def __init__(self, trunk, template, target_mean, target_std, coord_min,
                 coord_scale, hbc=None, cluster_m=0.7, n_vents_expected=3,
                 t_table=None, t_ref=294.0, vent_width_m=0.6):
        super().__init__()
        self.trunk = trunk
        self.template = template
        self.hbc = hbc                      # optional HardBCModel, used for its wall mask

        self.register_buffer("mu", target_mean[0:3].clone().float())
        self.register_buffer("sigma", target_std[0:3].clone().float())
        self.register_buffer("coord_min", coord_min.clone().float())
        self.register_buffer("coord_scale", coord_scale.clone().float())

        self.cluster_m = cluster_m
        self.n_vents_expected = n_vents_expected
        self.t_table = t_table
        self.t_ref = t_ref
        self.vent_width_m = vent_width_m

        self._cases = {}
        self._active = None

    # ------------------------------------------------------------------ #
    #  Case registration
    # ------------------------------------------------------------------ #
    @staticmethod
    def _fingerprint(pc_xyz):
        q = (pc_xyz.detach().cpu().double() * 1e6).round().to(torch.int64)
        return (int(q.shape[0]), int(q.sum().item()), int((q % 9973).sum().item()))

    def register_cases(self, dataset, verbose=True):
        """Extract vents for every room in `dataset` (including mirrored ones)."""
        n_new = 0
        for room in dataset.cases:
            key = self._fingerprint(room["pc_full"][:, :3])
            if key in self._cases:
                continue
            anchors, cond = extract_vents(
                room, self.coord_min.cpu(), self.coord_scale.cpu(),
                self.mu.cpu(), self.sigma.cpu(),
                cluster_m=self.cluster_m, n_expected=self.n_vents_expected,
                t_table=self.t_table, t_ref=self.t_ref,
                vent_width_m=self.vent_width_m)
            self._cases[key] = {"anchors": anchors, "cond": normalize_cond(cond),
                                "raw_cond": cond}
            n_new += 1
        if verbose and n_new:
            some = next(iter(self._cases.values()))
            print(f"[equijet] registered {n_new} case(s), K={len(some['anchors'])} "
                  f"vents each (cluster_m={self.cluster_m} m)")
        if self.hbc is not None and hasattr(self.hbc, "register_cases"):
            self.hbc.register_cases(dataset)
        return n_new

    def _resolve(self, pc):
        pc0 = pc[0] if pc.dim() == 3 else pc
        key = self._fingerprint(pc0[:, :3])
        entry = self._cases.get(key)
        if entry is None:
            raise KeyError(
                "no vents registered for this case. Call "
                "model.register_cases(dataset) for every dataset before training "
                "(the trainer subclass does this automatically).")
        dev = pc.device
        if entry.get("_dev") != dev:
            entry["anchors"] = entry["anchors"].to(dev)
            entry["cond"] = entry["cond"].to(dev)
            entry["_dev"] = dev
        return entry

    # ------------------------------------------------------------------ #
    #  Drop-in Trunk interface
    # ------------------------------------------------------------------ #
    def encode_geometry(self, pc, sample_ids=None):
        self._active = self._resolve(pc)
        if self.hbc is not None:
            self._active_wall = self.hbc._resolve(pc)
        return self.trunk.encode_geometry(pc, sample_ids=sample_ids)

    def decode_query(self, latent, xyt):
        base = self.trunk.decode_query(latent, xyt)
        return self._compose(base, xyt)

    def forward(self, xyt, pc, sample_ids=None):
        return self.decode_query(self.encode_geometry(pc, sample_ids), xyt)

    # ------------------------------------------------------------------ #
    #  The decomposition
    # ------------------------------------------------------------------ #
    def jets(self, xyt):
        """Summed template contribution at `xyt`, in PHYSICAL m/s. (N, 3)."""
        case = self._active
        x_flat = xyt.reshape(-1, xyt.shape[-1])
        x_phys = x_flat * self.coord_scale + self.coord_min          # (N, 3) metres

        anchors, cond = case["anchors"], case["cond"]                # (K,3), (K,C)
        xi = x_phys.unsqueeze(0) - anchors.unsqueeze(1)              # (K, N, 3)
        return self.template(xi, cond).sum(dim=0)                    # (N, 3)

    def _compose(self, base, xyt):
        """Combine template + correction (+ optional wall mask).

        Two paths, deliberately:

        * **No mask** — done in NORMALIZED space as `base + J/sigma`, which is
          algebraically identical to de-normalizing, adding J, and re-normalizing,
          but avoids the round-trip. That matters: `((b*s + m) - m)/s` is not
          bitwise `b` in floating point (measured 3e-8 drift), which would break
          the plan's guarantee that a zero-initialized template reproduces the
          baseline EXACTLY at epoch 0. With this form, J = 0 gives `base` back bit
          for bit.
        * **With mask** — the single physical-space path of plan §6.2: de-normalize
          once, add the jets, apply the wall mask to the SUM (the jet must vanish
          on the ceiling it hugs too), re-normalize once. No double round-trip.
        """
        shape = base.shape
        flat = base.reshape(-1, shape[-1])
        jets = self.jets(xyt)
        out = flat.clone()

        if self.hbc is None:
            out[:, 0:3] = flat[:, 0:3] + jets / self.sigma
            return out.reshape(shape)

        from hbc.distance import mask, min_dist
        u_phys = flat[:, 0:3] * self.sigma + self.mu + jets
        x_phys = xyt.reshape(-1, xyt.shape[-1]) * self.coord_scale + self.coord_min
        phi = min_dist(x_phys, self._active_wall["wall_phys"], self.hbc.chunk)
        u_phys = mask(phi, self.hbc.ell_wall).unsqueeze(-1) * u_phys

        out[:, 0:3] = (u_phys - self.mu) / self.sigma
        return out.reshape(shape)

    def describe(self):
        k = len(next(iter(self._cases.values()))["anchors"]) if self._cases else "?"
        return (f"EquiJetModel(K={k}, cluster_m={self.cluster_m} m, "
                f"s={float(self.template.s):.2f} m, "
                f"hbc={'ell=' + str(self.hbc.ell_wall) + ' m' if self.hbc else 'off'}, "
                f"cases={len(self._cases)})")
