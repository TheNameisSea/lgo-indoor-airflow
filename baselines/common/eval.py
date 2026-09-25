"""Shared evaluation for every model in the comparison.

This is a faithful port of `HybridGINOTTrainer.validate` (pi_ginot/trainer.py:48)
lifted out of the trainer so that GINOT and every baseline are scored by the SAME
code path. The only deliberate change is `seed`: the point subsample is drawn from
a fixed generator so all models are scored on the IDENTICAL points, which removes
subsample noise from the head-to-head comparison.

Any model in the comparison must expose the GINOT interface:
    encode_geometry(pc: (1, N_pc, C)) -> latent          (model-specific object)
    decode_query(latent, xyz: (1, M, 3)) -> (1, M, 3)    (u, v, w in normalized units)
"""

import torch


@torch.no_grad()
def evaluate(model, val_dataset, device, n_points=50000, chunk=20000, seed=0):
    """De-normalized full-field velocity metrics on held-out cases.

    Returns the same dict as the GINOT trainer: vel_mse, vel_relL2, vel_r2,
    vel_std_pred/true, vel_std_ratio (struct%), val_combined.
    """
    was_training = model.training
    model.eval()

    tmean = val_dataset.target_mean.to(device)
    tstd = val_dataset.target_std.to(device)

    agg = {"vel_mse": 0.0, "p_mse": 0.0, "vel_relL2": 0.0,
           "vel_std_pred": 0.0, "vel_std_true": 0.0, "vel_r2": 0.0}
    n_scored = 0

    # Fixed generator -> identical subsample for every model and every epoch.
    gen = torch.Generator().manual_seed(seed)

    # Transolver's slice statistics depend on the token set, so it pins the eval
    # chunk to its training query count; other models are chunk-invariant.
    chunk = getattr(model, "eval_chunk", None) or chunk

    for room in val_dataset.cases:
        xyz = torch.cat([room["pool_stream_xyz"], room["pool_bg_xyz"]], dim=0)
        tgt = torch.cat([room["pool_stream_tgt"], room["pool_bg_tgt"]], dim=0)
        if len(xyz) == 0:
            continue

        pc = room["pc_full"].unsqueeze(0).to(device)
        latent = model.encode_geometry(pc)

        if len(xyz) > n_points:
            sel = torch.randperm(len(xyz), generator=gen)[:n_points]
            xyz, tgt = xyz[sel], tgt[sel]
        tgt = tgt.to(device)

        preds = []
        for i in range(0, len(xyz), chunk):
            q = xyz[i:i + chunk].unsqueeze(0).to(device)
            preds.append(model.decode_query(latent, q).squeeze(0))
        pred = torch.cat(preds, dim=0)

        nch = pred.shape[-1]
        pred_phys = pred * tstd[:nch] + tmean[:nch]
        tgt_phys = tgt * tstd + tmean

        pred_vel, tgt_vel = pred_phys[:, 0:3], tgt_phys[:, 0:3]
        vel_mse = torch.mean((pred_vel - tgt_vel) ** 2).item()
        vel_relL2 = (torch.linalg.norm(pred_vel - tgt_vel)
                     / (torch.linalg.norm(tgt_vel) + 1e-8)).item()
        vel_var = tgt_vel.var(dim=0, unbiased=False).mean().item()
        vel_r2 = 1.0 - vel_mse / vel_var if vel_var > 1e-12 else 0.0

        agg["vel_mse"] += vel_mse
        agg["p_mse"] += float("nan")
        agg["vel_relL2"] += vel_relL2
        agg["vel_r2"] += vel_r2
        agg["vel_std_pred"] += pred_vel.std(dim=0).mean().item()
        agg["vel_std_true"] += tgt_vel.std(dim=0).mean().item()
        n_scored += 1

    for k in agg:
        agg[k] /= max(1, n_scored)
    agg["val_combined"] = agg["vel_mse"]
    agg["vel_std_ratio"] = agg["vel_std_pred"] / max(1e-12, agg["vel_std_true"])

    if was_training:
        model.train()
    return agg
