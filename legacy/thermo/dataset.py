"""ThermoDataset — MultiCase_GINOT_Dataset plus a temperature target channel.

A pure subclass. `pi_ginot/dataset.py` is NOT modified, so the velocity-only
baseline stays byte-identical for A/B, exactly as `hbc/` and `equijet/` do.

Targets become `[u, v, w, p, T]`. Temperature is **appended**, so every existing
index is unchanged (velocity 0:3, pressure 3) and old 4-channel checkpoints and
stats files remain valid.

Two corrections are applied after the parent finishes, both forced by how the raw
data is shaped:

1. **Interior-only temperature statistics.** Only `Fluid_data.csv` and
   `New_HVAC.csv` carry a temperature column; the loader fills every wall, leak
   and furniture point with 0.0. Pooling those zeros against a real ~294 K field
   destroys the mean and the std. This is exactly the "A4" problem the parent
   already documents for pressure, and it needs the same interior-only fix. The
   parent normalises the pools before we can intervene, so the correction
   de-normalises channel 4 with the bad statistics and re-normalises with the good
   ones — algebraically exact, no reload.

2. **Return-vent temperature is blinded.** Supply temperature is a design input
   and stays visible to the encoder. Return-vent temperature is the room's mixed
   mean air temperature — a RESULT of the solution, not a boundary condition
   (measured: 287.2 K supply vs 291.8 K return). Leaving it in the point cloud
   would hand the model most of the temperature answer. Wall and leak temperatures
   are blinded too, since they are loader-filled zeros rather than data.
"""

import os

import torch

from pi_ginot import MultiCase_GINOT_Dataset

T_COL = "Temperature (K)"
T_IDX = 4                     # target channel
PC_T_IDX = 3 + 5 + T_IDX      # 12: xyz(3) + class one-hot(5) + targets
PC_CLS = slice(3, 8)
CLS_INLET = 2

_POOLS = ("pool_stream_tgt", "pool_bg_tgt", "pool_wall_tgt",
          "pool_in_tgt", "pool_out_tgt", "pool_leak_tgt")


class ThermoDataset(MultiCase_GINOT_Dataset):
    def _build_dataset(self, n_case_pool, n_boundary_pool, saved_stats):
        # Inject the extra target column BEFORE the parent reads any CSV. The
        # parent already fills missing columns with 0.0, so surface files that
        # carry no temperature are handled without touching the loader.
        if T_COL not in self.cols_uvwp:
            self.cols_uvwp = self.cols_uvwp + [T_COL]
        if self.pc_mode != "geom":
            self.pc_channels = 3 + 5 + len(self.cols_uvwp)

        super()._build_dataset(n_case_pool, n_boundary_pool, saved_stats)

        if saved_stats is None:
            self._fix_temperature_stats()      # correction 1
        self._blind_non_supply_temperature()   # correction 2

    # ------------------------------------------------------------------ #
    def _fix_temperature_stats(self):
        """Recompute channel 4 statistics from interior points only."""
        bad_mean = self.target_mean[T_IDX].clone()
        bad_std = self.target_std[T_IDX].clone()

        interior = torch.cat([c[k][:, T_IDX] for c in self.cases
                              for k in ("pool_stream_tgt", "pool_bg_tgt")
                              if len(c[k])], dim=0)
        interior_phys = interior * bad_std + bad_mean
        good_mean = interior_phys.mean()
        good_std = interior_phys.std()
        if good_std < 1e-6:
            good_std = torch.tensor(1.0)

        # Exact re-normalisation: undo the bad transform, apply the good one.
        for case in self.cases:
            for k in _POOLS:
                t = case.get(k)
                if t is None or len(t) == 0:
                    continue
                t[:, T_IDX] = ((t[:, T_IDX] * bad_std + bad_mean) - good_mean) / good_std
            pc = case.get("pc_full")
            if pc is not None and pc.shape[1] > PC_T_IDX:
                pc[:, PC_T_IDX] = (
                    (pc[:, PC_T_IDX] * bad_std + bad_mean) - good_mean) / good_std

        self.target_mean[T_IDX] = good_mean
        self.target_std[T_IDX] = good_std

        # RE-SAVE. The parent writes stats_path during its own _build_dataset,
        # i.e. BEFORE this correction runs, so the file on disk holds the
        # uncorrected values. Any later dataset built with saved_stats=<that file>
        # then loads the bad statistics AND skips this fix, because the fix only
        # runs when stats are being computed. Symptom: the validation set is
        # normalised on a different scale from training, and de-normalised
        # predictions come out ~24x too large (measured s = 22.3 with rho = 0.94,
        # i.e. a perfectly good field on a wrong scale).
        if self.stats_path and os.path.exists(self.stats_path):
            blob = torch.load(self.stats_path, weights_only=True)
            blob["target_mean"] = self.target_mean
            blob["target_std"] = self.target_std
            torch.save(blob, self.stats_path)
            print(f"[thermo] re-saved corrected stats -> {self.stats_path}")

        print(f"[thermo] temperature channel (interior-only): "
              f"mean={good_mean:.2f} K, std={good_std:.4f} K   "
              f"(pooled-with-zeros would have been mean={bad_mean:.2f}, "
              f"std={bad_std:.4f})")

    def _blind_non_supply_temperature(self):
        """Keep supply-vent temperature; hide everything else from the encoder."""
        n_blind = n_keep = 0
        for case in self.cases:
            pc = case.get("pc_full")
            if pc is None or pc.shape[1] <= PC_T_IDX:
                continue
            is_inlet = pc[:, PC_CLS].argmax(dim=-1) == CLS_INLET
            # 0.0 in normalised space == the mean temperature, which is the same
            # convention the parent uses when it blinds pressure.
            pc[~is_inlet, PC_T_IDX] = 0.0
            n_blind += int((~is_inlet).sum())
            n_keep += int(is_inlet.sum())
        print(f"[thermo] point cloud: supply-vent temperature kept at {n_keep:,} "
              f"points, blinded at {n_blind:,} (return vents, walls, leaks)")
