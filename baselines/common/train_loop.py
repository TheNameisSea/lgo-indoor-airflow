"""Generic training loop shared by every baseline.

Mirrors `HybridGINOTTrainer.train` (pi_ginot/trainer.py:136) with everything the
comparison does not use stripped out: no PDE residual, no boundary losses, no
pressure. What remains is exactly what produced runs/ginot_pc_2000ep:

  * one case per forward, `accum_cases=1` -> one optimizer step per case
  * relative velocity loss (pi_ginot.losses.supervised_data_loss)
  * Adam + OneCycleLR (pct_start 0.3, div 25, final_div 1e4)
  * grad-norm clip 1.0
  * validation every `val_every` epochs, best.pth on max val vel_r2

Only interior case points (bc_mask == 0) are supervised. In the GINOT reference
run the boundary points were forwarded but multiplied by lambda_*=0, so this is
gradient-identical and simply skips wasted compute.
"""

import json
import os
import time

import torch

from pi_ginot.losses import supervised_data_loss
from .eval import evaluate


def train_model(model, train_ds, val_ds, device, out_dir, epochs=2000, lr=1e-4,
                lambda_data=10.0, vel_floor=0.2, val_every=25, ckpt_every=200,
                log_every=5, grad_clip=1.0, eval_seed=0):
    raise RuntimeError(
        "baselines/common/train_loop.py::train_model is NOT the training path "
        "and must not become one. It selects best.pth on max val vel_r2 (see "
        "below); every arm in this repo is trained through train.py -> "
        "train_thermo.main() -> ThermoTrainer, which selects on the joint "
        "TAYLOR score. That difference is not cosmetic: "
        "R2 = rho^2 - (rho-s)^2 - bias^2 is maximised at s = rho, so selecting "
        "on R2 REWARDS DAMPING -- it picks the checkpoint that under-predicts "
        "amplitude. Taylor = 2(1+rho)/(s+1/s)^2 needs correlation AND unit "
        "amplitude together. A baseline selected by this function against an "
        "incumbent selected by ThermoTrainer is not a comparison.\n\n"
        "Nothing has ever called this (verified: zero callers outside a "
        "docstring), so no published number is affected. It is kept only "
        "because the module docstring is the clearest record of what produced "
        "runs/ginot_pc_2000ep. Use train.py. If you genuinely need a "
        "standalone loop, port ThermoTrainer's selection rule first and delete "
        "this guard deliberately.")
    os.makedirs(out_dir, exist_ok=True)
    history = []

    loader = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True)
    n_cases = len(loader)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=lr, epochs=epochs, steps_per_epoch=n_cases,
        pct_start=0.3, div_factor=25.0, final_div_factor=1e4,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Training {epochs} epochs on {device} | {n_cases} cases/epoch | "
          f"{epochs * n_cases:,} optimizer updates | {n_params:,} trainable params",
          flush=True)

    best = float("-inf")
    best_epoch = -1
    t0 = time.time()

    for ep in range(epochs):
        log = {"data": 0.0, "vel_std": 0.0}
        n_seen = 0

        for batch in loader:
            pc = batch["pc"].to(device)
            xyt = batch["xyt"].to(device)
            targets = batch["targets"].to(device)
            bc_mask = batch["bc_mask"].to(device)
            target_mean = batch["target_mean"][0].to(device)
            target_std = batch["target_std"][0].to(device)

            # Interior case points only (mask 0); boundary masks carry no loss.
            # `is_stream` is built over the case points ALONE (n_colloc + n_super)
            # and mask 0 selects exactly those, in the same order — so it lines up
            # with q_tgt directly and must NOT be re-indexed by `keep`.
            keep = (bc_mask.squeeze(0) == 0)
            q_xyz = xyt[:, keep]
            q_tgt = targets[:, keep].squeeze(0)

            if q_xyz.shape[1] == 0:
                continue

            optim.zero_grad(set_to_none=True)

            latent = model.encode_geometry(pc)
            preds = model.decode_query(latent, q_xyz).squeeze(0)

            loss_data = supervised_data_loss(
                preds, q_tgt, lambda_p_boost=0.0, vel_floor=vel_floor,
                target_mean=target_mean, target_std=target_std)

            if loss_data is None or not torch.isfinite(loss_data):
                scheduler.step()
                continue

            (lambda_data * loss_data).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optim.step()
            scheduler.step()

            log["data"] += loss_data.item()
            with torch.no_grad():
                v_phys = preds[:, 0:3] * target_std[0:3] + target_mean[0:3]
                log["vel_std"] += v_phys.std(dim=0).mean().item()
            n_seen += 1

            del preds, latent

        for k in log:
            log[k] /= max(1, n_seen)

        if ep == 0 or (ep + 1) % log_every == 0 or ep == epochs - 1:
            cur_lr = optim.param_groups[0]["lr"]
            print({"epoch": ep + 1, "data": f"{log['data']:.4e}",
                   "lr": f"{cur_lr:.2e}", "vel_std": f"{log['vel_std']:.4f}",
                   "min": f"{(time.time() - t0) / 60:.1f}"}, flush=True)
            history.append({"epoch": ep + 1, "data": log["data"], "lr": cur_lr,
                            "vel_std": log["vel_std"]})

        do_val = val_ds is not None and ((ep + 1) % val_every == 0 or ep == epochs - 1)
        if do_val:
            metrics = evaluate(model, val_ds, device, seed=eval_seed)
            print(f"  [val] epoch {ep + 1}: vel_mse={metrics['vel_mse']:.4e} "
                  f"vel_R2={metrics['vel_r2']:.4f} "
                  f"relL2={metrics['vel_relL2']:.4f} "
                  f"struct={100 * metrics['vel_std_ratio']:.1f}%", flush=True)
            if history and history[-1]["epoch"] == ep + 1:
                history[-1]["val"] = metrics
            else:
                history.append({"epoch": ep + 1, "val": metrics})
            if metrics["vel_r2"] > best:
                best = metrics["vel_r2"]
                best_epoch = ep + 1
                torch.save(model.state_dict(), os.path.join(out_dir, "best.pth"))
                print(f"  [ckpt] new best vel_r2={best:.4f} -> best.pth", flush=True)

        if (ep + 1) % ckpt_every == 0:
            torch.save(model.state_dict(), os.path.join(out_dir, f"epoch_{ep + 1:05d}.pth"))

        torch.save(model.state_dict(), os.path.join(out_dir, "last.pth"))
        with open(os.path.join(out_dir, "history.json"), "w") as fh:
            json.dump(history, fh, indent=2)

    summary = {"best_vel_r2": best, "best_epoch": best_epoch,
               "n_params": n_params, "minutes": (time.time() - t0) / 60,
               "epochs": epochs}
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Done. best vel_r2={best:.4f} @ epoch {best_epoch} "
          f"({summary['minutes']:.1f} min). -> {out_dir}", flush=True)
    return summary
