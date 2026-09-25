"""
Training loop for PI-GINOT.

Rewrite of notebook cell 10 applying audit fixes:

  * Fix 2 (gradient accumulation): one optimizer step per epoch over ALL cases,
    instead of one noisy step per room. Gradients are averaged across the 9 cases.
  * Fix 3 (debug): the per-batch NaN "trap" forwards / dead code are removed (a
    lightweight finite guard and an optional --debug remain).
  * Fix 5 (validation + checkpointing): periodic validation on held-out cases with
    de-normalized full-field MSE, and best/last/periodic checkpoints.

Losses come from pi_ginot.losses. This trainer is DATA-ONLY: the PDE-residual
avenue was closed by three controlled negatives (PROJECT_OVERVIEW.md) -- momentum
was poisoned by ~0-skill pressure and continuity was redundant with data -- so the
residual machinery has been removed rather than left as a dormant flag.
"""

import json
import os

import torch

from .losses import (
    wall_bc_loss,
    inlet_bc_loss,
    outlet_bc_loss,
    leak_bc_loss,
    supervised_data_loss,
    center_pressure,
)


class HybridGINOTTrainer:
    def __init__(self, model, device, out_dir="./runs", debug=False):
        self.model = model.to(device)
        self.device = device
        self.out_dir = out_dir
        self.debug = debug
        os.makedirs(out_dir, exist_ok=True)
        self.history = []

    # ------------------------------------------------------------------ #
    #  Validation
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def validate(self, val_dataset, n_points=50000, chunk=20000):
        """De-normalized full-field MSE / relative-L2 on held-out cases.

        Returns a dict of averaged metrics; `val_combined` (velocity MSE + pressure
        MSE, both in physical units) is the scalar used for best-model selection.
        """
        self.model.eval()
        tmean = val_dataset.target_mean.to(self.device)
        tstd = val_dataset.target_std.to(self.device)

        agg = {"vel_mse": 0.0, "p_mse": 0.0, "vel_relL2": 0.0,
               "vel_std_pred": 0.0, "vel_std_true": 0.0, "vel_r2": 0.0}
        n_scored = 0

        for room in val_dataset.cases:
            xyz = torch.cat([room["pool_stream_xyz"], room["pool_bg_xyz"]], dim=0)
            tgt = torch.cat([room["pool_stream_tgt"], room["pool_bg_tgt"]], dim=0)
            if len(xyz) == 0:
                continue  # no interior points to score

            pc = room["pc_full"].unsqueeze(0).to(self.device)
            latent = self.model.encode_geometry(pc)

            if len(xyz) > n_points:
                sel = torch.randperm(len(xyz))[:n_points]
                xyz, tgt = xyz[sel], tgt[sel]
            tgt = tgt.to(self.device)

            preds = []
            for i in range(0, len(xyz), chunk):
                q = xyz[i:i + chunk].unsqueeze(0).to(self.device)
                preds.append(self.model.decode_query(latent, q).squeeze(0))
            pred = torch.cat(preds, dim=0)

            # De-normalize to physical units. The model may output fewer channels
            # than the 4-wide (u,v,w,p) stats, so slice the stats to match.
            nch = pred.shape[-1]
            pred_phys = pred * tstd[:nch] + tmean[:nch]
            tgt_phys = tgt * tstd + tmean

            pred_vel, tgt_vel = pred_phys[:, 0:3], tgt_phys[:, 0:3]
            vel_mse = torch.mean((pred_vel - tgt_vel) ** 2).item()
            vel_relL2 = (torch.linalg.norm(pred_vel - tgt_vel)
                         / (torch.linalg.norm(tgt_vel) + 1e-8)).item()
            # Per-case velocity R^2 (zero-skill = the case's own field variance).
            vel_var = tgt_vel.var(dim=0, unbiased=False).mean().item()
            vel_r2 = 1.0 - vel_mse / vel_var if vel_var > 1e-12 else 0.0

            # Pressure centered per case (gauge-invariant); only if predicted.
            if nch >= 4:
                pp, tp = center_pressure(pred_phys[:, 3], tgt_phys[:, 3])
                p_mse = torch.mean((pp - tp) ** 2).item()
            else:
                p_mse = float("nan")

            agg["vel_mse"] += vel_mse
            agg["p_mse"] += p_mse
            agg["vel_relL2"] += vel_relL2
            agg["vel_r2"] += vel_r2
            # Structure amplitude, measured PRED vs TRUE on the SAME points.
            # Self-referencing on purpose: an external constant is easy to get
            # wrong (it must match the split and the sampler's stream/bg mix).
            agg["vel_std_pred"] += pred_vel.std(dim=0).mean().item()
            agg["vel_std_true"] += tgt_vel.std(dim=0).mean().item()
            n_scored += 1

        for k in agg:
            agg[k] /= max(1, n_scored)
        p_term = agg["p_mse"] if agg["p_mse"] == agg["p_mse"] else 0.0  # nan-safe
        agg["val_combined"] = agg["vel_mse"] + p_term
        # 1.0 = correct structure amplitude; <1 = over-smoothed.
        agg["vel_std_ratio"] = agg["vel_std_pred"] / max(1e-12, agg["vel_std_true"])
        self.model.train()
        return agg

    # ------------------------------------------------------------------ #
    #  Checkpointing
    # ------------------------------------------------------------------ #
    def _save(self, name):
        torch.save(self.model.state_dict(), os.path.join(self.out_dir, name))

    def _write_history(self):
        with open(os.path.join(self.out_dir, "history.json"), "w") as fh:
            json.dump(self.history, fh, indent=2)

    # ------------------------------------------------------------------ #
    #  Training
    # ------------------------------------------------------------------ #
    def train(self, dataset, val_dataset=None, epochs=3600, lr=1e-4,
              lambda_data=10.0, lambda_wall=10.0, lambda_inlet=10.0,
              lambda_outlet=10.0, lambda_leak=1.0, lambda_p=1.0, vel_floor=0.1,
              val_every=25, ckpt_every=200, log_every=5,
              ckpt_metric="vel_mse", accum_cases=1):
        """Data-only training loop.

        `ckpt_metric` selects best.pth. Default "vel_mse", NOT "val_combined":
        combined is ~99.7% pressure on this data, so selecting on it picks
        checkpoints by pressure error and ignores the airflow task.

        `accum_cases` = how many cases to accumulate before an optimizer step.
        This controls the OPTIMIZER STEP BUDGET, which is easy to starve:
        accumulating over all cases gives 1 step/epoch, so 2000 epochs = 2000
        updates -- versus 32,400 in the original per-case setup (3600 epochs x 9
        cases). Under-stepping looks exactly like under-fitting. Forward/backward
        cost is identical either way (only optim.step() count changes), so a
        SMALL value buys many more updates for the same compute.
          1              -> per-case steps, max updates (matches the original)
          n_cases        -> one full-dataset step per epoch (cleanest gradient)
        """
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
        n_cases = len(loader)
        optim = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Optimizer-step budget: ceil(n_cases / accum_cases) steps per epoch.
        accum_cases = max(1, min(int(accum_cases), n_cases))
        steps_per_epoch = (n_cases + accum_cases - 1) // accum_cases
        # Group boundaries so a short final group is weighted by its true size.
        group_sizes = [min(accum_cases, n_cases - s)
                       for s in range(0, n_cases, accum_cases)]

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optim, max_lr=lr, epochs=epochs, steps_per_epoch=steps_per_epoch,
            pct_start=0.3, div_factor=25.0, final_div_factor=1e4,
        )

        print(f"Starting training for {epochs} epochs on {self.device} "
              f"({n_cases} cases/epoch, accum_cases={accum_cases} -> "
              f"{steps_per_epoch} step(s)/epoch, {epochs * steps_per_epoch:,} "
              f"optimizer updates total).")
        print(f"Checkpointing best.pth on '{ckpt_metric}'.")

        # vel_r2 is a "higher is better" metric; the rest are minimized.
        maximize = ckpt_metric in ("vel_r2",)
        best = float("-inf") if maximize else float("inf")

        for ep in range(epochs):
            # Per-epoch accumulators (averaged over cases for honest logging).
            log = {"data": 0.0, "wall": 0.0, "inlet": 0.0, "outlet": 0.0,
                   "leak": 0.0, "vel_std": 0.0}

            optim.zero_grad()
            group_idx, in_group = 0, 0

            for batch in loader:
                pc = batch["pc"].to(self.device)
                xyt = batch["xyt"].to(self.device)
                targets = batch["targets"].to(self.device)
                bc_mask = batch["bc_mask"].to(self.device)

                coord_scale = batch["coord_scale"][0].to(self.device)
                target_mean = batch["target_mean"][0].to(self.device)
                target_std = batch["target_std"][0].to(self.device)
                leak_mag = batch["leak_mag"].squeeze(0).to(self.device)

                if self.debug and (torch.isnan(pc).any() or torch.isnan(xyt).any()
                                   or torch.isnan(targets).any()):
                    print("[debug] NaN in raw batch; skipping case.")
                    continue

                latent = self.model.encode_geometry(pc)

                case_loss = 0.0
                loss_data_raw = None

                # ---------- Supervised + boundary losses ----------
                # C1: supervise ALL interior points, including the collocation
                # ones -- they carry mask==0 and real CFD targets.
                data_xyt, data_mask, data_targets = xyt, bc_mask, targets

                preds = self.model.decode_query(latent, data_xyt).squeeze(0)
                true_targets = data_targets.squeeze(0)
                masks = data_mask.squeeze(0)

                if lambda_data != 0.0:
                    loss_data_raw = supervised_data_loss(
                        preds[masks == 0], true_targets[masks == 0],
                        lambda_p_boost=lambda_p, vel_floor=vel_floor,
                        target_mean=target_mean, target_std=target_std)
                    if loss_data_raw is not None:
                        log["data"] += loss_data_raw.item()

                if lambda_wall != 0.0:
                    loss_wall = wall_bc_loss(preds[masks == 1], true_targets[masks == 1])
                    case_loss = case_loss + (lambda_wall * loss_wall)
                    log["wall"] += loss_wall.item()

                if lambda_inlet != 0.0:
                    loss_inlet = inlet_bc_loss(preds[masks == 2], true_targets[masks == 2])
                    case_loss = case_loss + (lambda_inlet * loss_inlet)
                    log["inlet"] += loss_inlet.item()

                if lambda_outlet != 0.0:
                    loss_outlet = outlet_bc_loss(preds[masks == 3], true_targets[masks == 3],
                                                 is_active_exhaust=True)
                    case_loss = case_loss + (lambda_outlet * loss_outlet)
                    log["outlet"] += loss_outlet.item()

                # C2: direction-free leak constraint (mask 4 previously had NO loss,
                # so those query points were forwarded for zero gradient).
                if lambda_leak != 0.0:
                    loss_leak = leak_bc_loss(preds[masks == 4], leak_mag,
                                             target_mean, target_std)
                    case_loss = case_loss + (lambda_leak * loss_leak)
                    log["leak"] += loss_leak.item()

                # C4: collapse diagnostic -- spatial std of predicted interior
                # velocity in physical units. Craters if the field goes uniform.
                with torch.no_grad():
                    interior = preds[masks == 0]
                    if interior.numel() > 0:
                        v_phys = interior[:, 0:3] * target_std[0:3] + target_mean[0:3]
                        log["vel_std"] += v_phys.std(dim=0).mean().item()

                if loss_data_raw is not None:
                    case_loss = case_loss + (lambda_data * loss_data_raw)

                # Average the gradient over this accumulation group.
                gsize = group_sizes[group_idx]
                if isinstance(case_loss, torch.Tensor):
                    if torch.isfinite(case_loss):
                        (case_loss / gsize).backward()
                    elif self.debug:
                        print("[debug] non-finite case loss; skipping backward.")

                del preds, true_targets

                # ---------- Step once the group is complete ----------
                in_group += 1
                if in_group == gsize:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optim.step()
                    scheduler.step()
                    optim.zero_grad()
                    group_idx += 1
                    in_group = 0

            # Average per-case logs.
            for k in log:
                log[k] /= max(1, n_cases)

            # ---------- Logging ----------
            if ep == 0 or (ep + 1) % log_every == 0 or ep == epochs - 1:
                total = (lambda_data * log["data"]
                         + lambda_wall * log["wall"] + lambda_inlet * log["inlet"]
                         + lambda_outlet * log["outlet"] + lambda_leak * log["leak"])
                cur_lr = optim.param_groups[0]["lr"]
                msg = {"epoch": ep + 1, "total": f"{total:.4e}", "lr": f"{cur_lr:.2e}",
                       "data": f"{log['data']:.4e}", "wall": f"{log['wall']:.4e}",
                       "inlet": f"{log['inlet']:.4e}", "outlet": f"{log['outlet']:.4e}",
                       "leak": f"{log['leak']:.4e}", "vel_std": f"{log['vel_std']:.4f}"}
                print(msg)
                self.history.append({"epoch": ep + 1, "total": total, "lr": cur_lr,
                                     **log})

            # ---------- Validation + checkpoint ----------
            do_val = val_dataset is not None and ((ep + 1) % val_every == 0 or ep == epochs - 1)
            if do_val:
                metrics = self.validate(val_dataset)
                print(f"  [val] epoch {ep + 1}: vel_mse={metrics['vel_mse']:.4e} "
                      f"vel_R2={metrics['vel_r2']:.4f} "
                      f"p_mse={metrics['p_mse']:.4e} vel_relL2={metrics['vel_relL2']:.4f} "
                      f"vel_std={metrics['vel_std_pred']:.3f}/{metrics['vel_std_true']:.3f}"
                      f"={100 * metrics['vel_std_ratio']:.1f}%")
                if self.history and self.history[-1]["epoch"] == ep + 1:
                    self.history[-1]["val"] = metrics
                score = metrics[ckpt_metric]
                if (score > best) if maximize else (score < best):
                    best = score
                    self._save("best.pth")
                    print(f"  [ckpt] new best {ckpt_metric}={best:.4e} -> best.pth")

            if (ep + 1) % ckpt_every == 0:
                self._save(f"epoch_{ep + 1:05d}.pth")

            self._save("last.pth")
            self._write_history()

        # Fallback: if no validation was used, best.pth is the final model.
        if val_dataset is None:
            self._save("best.pth")
        print(f"Training complete. Checkpoints in {self.out_dir} (best {ckpt_metric}={best:.4e}).")
        return self.history
