"""
Boundary-condition losses and the per-case zero-mean pressure helper.

Ported from notebook cell 9. Fix 4 (pressure gauge freedom): incompressible
pressure is only defined up to an additive constant per case, so we subtract each
tensor's own mean before comparing pressure. With batch_size == 1 (one room per
forward), the mean over the point set is exactly the per-case gauge, which removes
the spurious constant offset seen in the previous test plots.
"""

import torch
import torch.nn.functional as F


def center_pressure(pred_p, targ_p):
    """Subtract each tensor's own mean before pressure comparison (per-case gauge).

    Both inputs are 1D (N,) pressure vectors for a single case. Returns the
    zero-mean prediction and target so an MSE between them is gauge-invariant.
    """
    if pred_p.numel() == 0:
        return pred_p, targ_p
    return pred_p - pred_p.mean(), targ_p - targ_p.mean()


def wall_bc_loss(pred_wall, target_wall):
    """No-slip wall condition (u, v, w = 0), matched in normalized space."""
    if pred_wall.numel() == 0:
        return torch.tensor(0.0, device=pred_wall.device, requires_grad=True)
    return F.mse_loss(pred_wall[:, :3], target_wall[:, :3])


def inlet_bc_loss(pred_inlet, target_inlet):
    """Inlet velocity drive (u, v, w). Pressure is a downstream result, not enforced."""
    if pred_inlet.numel() == 0:
        return torch.tensor(0.0, device=pred_inlet.device, requires_grad=True)
    return F.mse_loss(pred_inlet[:, :3], target_inlet[:, :3])


def outlet_bc_loss(pred_outlet, target_outlet, is_active_exhaust=False):
    """Outlet condition.

    Enforces pressure (gauge-centered per case) always, plus the vertical exhaust
    velocity when the outlet is an active exhaust fan.
    """
    if pred_outlet.numel() == 0:
        return torch.tensor(0.0, device=pred_outlet.device, requires_grad=True)

    # Pressure (index 3), centered per case -- only if the model outputs pressure.
    if pred_outlet.shape[1] >= 4:
        pred_p, targ_p = center_pressure(pred_outlet[:, 3], target_outlet[:, 3])
        loss_p = F.mse_loss(pred_p, targ_p)
    else:
        loss_p = torch.tensor(0.0, device=pred_outlet.device, requires_grad=True)

    if is_active_exhaust:
        loss_w = F.mse_loss(pred_outlet[:, 2:3], target_outlet[:, 2:3])
        return loss_p + loss_w
    return loss_p


def leak_bc_loss(pred_leak, leak_mag, target_mean, target_std):
    """Direction-free leak constraint: match the predicted SPEED to the measured one.

    `Leak.csv` reports only `Velocity: Magnitude (m/s)` -- no components. Assigning
    that magnitude to a single velocity component (as the original code attempted)
    would fabricate a direction, so instead we compare ||u_pred|| against the
    measured magnitude. Prediction is de-normalized to physical m/s first, because
    the magnitude is a physical quantity and does not share the per-component
    z-score stats.
    """
    if pred_leak.numel() == 0 or leak_mag.numel() == 0:
        return torch.tensor(0.0, device=pred_leak.device, requires_grad=True)

    vel_phys = pred_leak[:, 0:3] * target_std[0:3] + target_mean[0:3]
    mag_pred = torch.sqrt((vel_phys ** 2).sum(dim=-1) + 1e-12)
    return F.mse_loss(mag_pred, leak_mag)


def supervised_data_loss(pred_sup, targ_sup, lambda_p_boost=1.0, vel_floor=0.1,
                         target_mean=None, target_std=None):
    """Velocity + gauge-centered pressure loss, squared RELATIVE error.

    weight = 1 / (|v_phys| + vel_floor)^2, i.e. a 20% error on a 0.15 m/s wall jet
    counts like a 20% error on a 5 m/s vent jet. `vel_floor` [m/s] stops it
    exploding as |v| -> 0; it sets the speed below which errors stop being
    penalised harder.

    This replaced magnitude weighting, which combined with MSE's own v^2 scaling
    made fast regions ~2-3 ORDERS OF MAGNITUDE more important than slow ones, so
    the quiet bulk of the room contributed almost nothing to the gradient and was
    never learned (measured: structure amplitude ~10% at z=0.6 m vs ~37% near the
    vents). The magnitude/uniform modes and the explicit stream/background
    weighting were removed once relative weighting won outright.

    `target_mean`/`target_std` (4,) are needed to recover physical m/s.
    Returns a scalar loss, or None if there are no supervised points.
    """
    if pred_sup.numel() == 0:
        return None

    pred_vel, targ_vel = pred_sup[:, 0:3], targ_sup[:, 0:3]
    has_p = pred_sup.shape[1] >= 4        # model may output velocity only (u,v,w)

    # Per-point squared velocity error (summed over the 3 components).
    sq = ((pred_vel - targ_vel) ** 2).sum(dim=-1)

    if target_std is None:
        raise ValueError("supervised_data_loss needs target_mean/target_std")
    # Recover physical m/s to make `vel_floor` a meaningful speed.
    mean3 = target_mean[0:3] if target_mean is not None else 0.0
    phys = targ_vel * target_std[0:3] + mean3
    mag_p = torch.linalg.norm(phys, dim=-1).detach()
    # Errors are also in normalized units; rescale so the ratio is physical.
    scale = target_std[0:3].mean()
    w = (scale ** 2) / (mag_p + vel_floor) ** 2

    loss_vel = (sq * w).mean()

    # Pressure loss (gauge-centered per case). Skipped if the model does not
    # output pressure, or if lambda_p_boost is 0.
    if has_p and lambda_p_boost != 0.0:
        pred_p_c, targ_p_c = center_pressure(pred_sup[:, 3], targ_sup[:, 3])
        return loss_vel + lambda_p_boost * F.mse_loss(pred_p_c, targ_p_c)
    return loss_vel
