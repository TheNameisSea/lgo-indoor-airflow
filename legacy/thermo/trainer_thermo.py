"""ThermoTrainer — joint velocity + temperature training and selection.

`pi_ginot/losses.py` and `pi_ginot/trainer.py` are NOT modified. The trainer
imports `supervised_data_loss` into its own module namespace, so the temperature
term is added by rebinding that name for the duration of training — the same
"wrap, never edit" rule `hbc/` and `equijet/` follow.

Two things this changes versus the velocity-only trainer:

* **The loss** gains `lambda_t * MSE(T)`. Velocity uses a relative error weighted
  by 1/(|v|+floor)^2; temperature is a plain MSE on the z-scored channel, so both
  terms are O(1) and dimensionless and `lambda_t` starts at 1.0.

* **Checkpoint selection** is on the joint Taylor score, not `vel_r2`. R2 is
  maximised by under-predicting whenever correlation is low (see
  `thermo/metrics_joint.py`), so selecting on it would systematically prefer the
  most damped epoch. The Taylor score cannot be gamed that way, and averaging the
  velocity and temperature scores keeps one output from being sacrificed for the
  other.
"""

import torch
import torch.nn.functional as F

import pi_ginot.trainer as _T
from pi_ginot.losses import supervised_data_loss as _base_loss
from pi_ginot.trainer import HybridGINOTTrainer

from .metrics_joint import T_IDX, fmt, joint_metrics, summarise


def make_loss(lambda_t):
    def _loss(pred_sup, targ_sup, **kw):
        base = _base_loss(pred_sup, targ_sup, **kw)
        if base is None or pred_sup.shape[-1] <= T_IDX or lambda_t == 0.0:
            return base
        return base + lambda_t * F.mse_loss(pred_sup[:, T_IDX], targ_sup[:, T_IDX])
    return _loss


class ThermoTrainer(HybridGINOTTrainer):
    def __init__(self, *a, lambda_t=1.0, **kw):
        super().__init__(*a, **kw)
        self.lambda_t = float(lambda_t)

    @torch.no_grad()
    def validate(self, val_dataset, n_points=50000, chunk=20000):
        """Joint metrics on held-out cases, de-normalised to m/s and K."""
        was_training = self.model.training
        self.model.eval()
        tmean = val_dataset.target_mean.to(self.device)
        tstd = val_dataset.target_std.to(self.device)
        gen = torch.Generator().manual_seed(0)     # same points every epoch

        per_case = []
        for room in val_dataset.cases:
            xyz = torch.cat([room["pool_stream_xyz"], room["pool_bg_xyz"]], dim=0)
            tgt = torch.cat([room["pool_stream_tgt"], room["pool_bg_tgt"]], dim=0)
            if len(xyz) == 0:
                continue
            pc = room["pc_full"].unsqueeze(0).to(self.device)
            latent = self.model.encode_geometry(pc)
            if len(xyz) > n_points:
                sel = torch.randperm(len(xyz), generator=gen)[:n_points]
                xyz, tgt = xyz[sel], tgt[sel]
            pred = torch.cat([self.model.decode_query(
                latent, xyz[i:i + chunk].unsqueeze(0).to(self.device)).squeeze(0)
                for i in range(0, len(xyz), chunk)], dim=0)
            nch = pred.shape[-1]
            per_case.append(joint_metrics(
                (tgt.to(self.device) * tstd + tmean).cpu().numpy(),
                (pred * tstd[:nch] + tmean[:nch]).cpu().numpy()))

        m = summarise(per_case) if per_case else {"score": float("-inf")}
        # The base trainer's logging line reads a fixed set of keys, and its
        # checkpointer reads `ckpt_metric`. Supply both so nothing downstream has
        # to know this is a joint model.
        m["val_combined"] = -m["score"]      # minimise-key => maximises the score
        m["vel_std_ratio"] = m.get("vel_s", 0.0)
        m["vel_std_pred"] = m.get("vel_std_pred", 0.0)
        m["vel_std_true"] = m.get("vel_std_true", 0.0)
        m["vel_relL2"] = m.get("vel_relL2", 0.0)
        m["p_mse"] = float("nan")            # pressure is carried but unsupervised
        if was_training:
            self.model.train()
        return m

    def train(self, *a, **kw):
        # "val_combined", not "score". The base trainer decides direction by name
        # (`maximize = ckpt_metric in ("vel_r2",)`), so any new key would be
        # MINIMISED and would select the worst epoch. `val_combined` is already a
        # minimise-key, and validate() sets it to -score, so minimising it
        # maximises the joint Taylor score. No core file is touched.
        kw.setdefault("ckpt_metric", "val_combined")
        print(f"[thermo] joint velocity + temperature, lambda_t={self.lambda_t}, "
              f"selecting on max joint Taylor score (via -score in "
              f"'{kw['ckpt_metric']}')")
        original = _T.supervised_data_loss
        _T.supervised_data_loss = make_loss(self.lambda_t)
        try:
            return super().train(*a, **kw)
        finally:
            _T.supervised_data_loss = original     # never leave the patch in place
