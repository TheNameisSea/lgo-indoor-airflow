"""Joint velocity + temperature scoring, on the metrics the project moved to.

One place, used by every harness, so a GINOT variant and a Tier-1 baseline cannot
be scored differently.

WHY NOT R2. For a centred field `R2 = rho^2 - (rho - s)^2 - bias^2`, so R2 is
maximised at `s = rho` — i.e. by UNDER-predicting whenever correlation is low.
Measured on Case_16 at z = 2.80 m, GINOT+HBC took the best R2 (+0.260) with the
worst pattern skill of the leading models (rho 0.522 against 0.596), purely by
predicting 62% of the true amplitude. Ranking on R2 therefore rewards a model for
giving up, which is why selection here uses the Taylor score instead.

    rho     pattern skill; the ceiling on R2 (max R2 = rho^2). Damping cannot
            improve it.
    s       amplitude ratio, target 1. A calibration number, not a target to
            maximise (and NOT struct%, which implies bigger is better).
    Taylor  2(1+rho)/(s + 1/s)^2 — Taylor (2001). Needs BOTH high rho and s ~ 1.
            The standard remedy in climate-model evaluation for exactly this
            failure mode.
    mae     dimensional error: m/s for velocity, K for temperature. Readable
            against the ~0.15-0.25 m/s draught threshold and the ~0.5 K
            resolution of thermal-comfort work.
"""

import numpy as np
import torch

T_IDX = 4


def _stats(true, pred):
    """rho, s, Taylor, mae, R2 for an (N, C) field. Torch or numpy."""
    if isinstance(true, torch.Tensor):
        true = true.detach().cpu().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    true = np.asarray(true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if true.ndim == 1:
        true, pred = true[:, None], pred[:, None]

    tm, pm = true.mean(0), pred.mean(0)
    tc, pc = true - tm, pred - pm
    st = float(np.sqrt((tc ** 2).sum(1).mean()))
    sp = float(np.sqrt((pc ** 2).sum(1).mean()))
    rho = float((tc * pc).sum(1).mean() / (st * sp)) if st > 1e-12 and sp > 1e-12 else 0.0
    s = sp / st if st > 1e-12 else 0.0
    taylor = 2.0 * (1.0 + rho) / (s + 1.0 / s) ** 2 if s > 1e-9 else 0.0
    var = float(true.var(0).mean())
    mse = float(((pred - true) ** 2).mean())
    return {"rho": rho, "s": s, "taylor": taylor,
            "mae": float(np.linalg.norm(pred - true, axis=1).mean()),
            "r2": 1.0 - mse / var if var > 1e-12 else float("nan"),
            "mse": mse, "std_pred": sp, "std_true": st,
            "relL2": float(np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-8))}


def joint_metrics(true_phys, pred_phys, has_temperature=True):
    """Score a de-normalised prediction. `*_phys` are (N, C) in m/s and K.

    Returns a flat dict prefixed `vel_` / `temp_`, plus the selection scalar
    `score` = mean of the two Taylor scores (velocity only, if C < 5).
    """
    out = {f"vel_{k}": v for k, v in _stats(true_phys[:, 0:3], pred_phys[:, 0:3]).items()}
    if has_temperature and true_phys.shape[1] > T_IDX and pred_phys.shape[1] > T_IDX:
        out.update({f"temp_{k}": v for k, v in
                    _stats(true_phys[:, T_IDX], pred_phys[:, T_IDX]).items()})
        # Both outputs matter, and neither should be sacrificed for the other, so
        # selection uses the mean rather than a weighted sum -- a weight here would
        # silently re-introduce the lambda_t choice at checkpoint time.
        out["score"] = 0.5 * (out["vel_taylor"] + out["temp_taylor"])
    else:
        out["score"] = out["vel_taylor"]
    return out


def summarise(per_case):
    """Mean over cases of a list of `joint_metrics` dicts."""
    keys = sorted({k for d in per_case for k in d})
    return {k: float(np.mean([d[k] for d in per_case if k in d])) for k in keys}


def fmt(m):
    v = (f"VEL rho={m['vel_rho']:+.3f} s={m['vel_s']:.2f} S={m['vel_taylor']:.3f} "
         f"R2={m['vel_r2']:+.3f} mae={m['vel_mae']:.3f}m/s")
    if "temp_rho" in m:
        v += (f" | TEMP rho={m['temp_rho']:+.3f} s={m['temp_s']:.2f} "
              f"S={m['temp_taylor']:.3f} R2={m['temp_r2']:+.3f} "
              f"mae={m['temp_mae']:.3f}K")
    return v + f" | score={m['score']:.4f}"
