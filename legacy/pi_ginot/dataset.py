"""
MultiCase GINOT dataset.

Ported from notebook cell 8, with the data-audit fixes:

  * A1: `dropna` is restricted to the columns actually used. `NusseltNumber`
    carries a DBL_MAX sentinel on ~80% of interior rows; a blanket dropna (after
    the inf->nan replace) could silently delete most of the interior data.
  * A3: leak magnitude. `Leak.csv` only has a velocity MAGNITUDE (no components),
    so it is kept as a separate scalar for a direction-free loss instead of being
    fabricated into a velocity component.
  * A4: pressure gauge + stats. Incompressible pressure is defined only up to an
    arbitrary per-case constant, and the wall/leak files carry FAKE zero
    pressures (the column does not exist there). Pooling raw pressures across
    cases made target_std[3] measure BETWEEN-case gauge offsets (tens of Pa)
    instead of real pressure structure (~4 Pa within a case), which (a) amplified
    every normalized error at de-normalization and (b) shrank each case's real
    pressure signal to ~0.1 normalized units, making it invisible to the loss.
    Fix: subtract each case's interior mean pressure at load time (Fluid_data,
    inlet and outlet share the solver's gauge, so all three are shifted by the
    same constant), and compute the pressure channel's mean/std from INTERIOR
    points only -- fake wall/leak zeros never enter the pressure stats.

Normalization stats (and `rho_ref`) are computed once from the training cases and
saved to `stats_path`; a validation dataset is built by passing
`saved_stats=<that path>` so it reuses the exact training normalization.
"""

import glob
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MultiCase_GINOT_Dataset(Dataset):
    def __init__(self, case_dirs, saved_stats=None,
                 n_case_pool=300000, n_boundary_pool=100000, n_pc=15000,
                 n_collocation_batch=8192, n_supervised_batch=100,
                 n_boundary_batch=4096, n_inlet_batch=800,
                 n_outlet_batch=800, n_leak_batch=800,
                 stats_path="ginot_baseline_stats.pt", g=9.81,
                 pressure_gauge=True, augment=None, pc_mode="full",
                 hvac_patch_cluster_m=0.7):
        super().__init__()
        # A5: single-linkage distance [m] used to recover HVAC patches before
        # classifying each by its net flux. 0.7 m sits inside the ~1.5 m gap
        # between supply and return here; 0.0 restores the original per-point sign
        # rule. See the A5 note in _load_raw_room.
        self.hvac_patch_cluster_m = float(hvac_patch_cluster_m)

        self.case_dirs = case_dirs
        self.n_pc = n_pc
        self.stats_path = stats_path
        self.g = g
        # A4: remove each case's arbitrary pressure gauge and take the pressure
        # stats from interior points only. Set False to restore the original
        # pooled statistics (which mixed in fake wall/leak zeros and cross-case
        # gauge offsets). NOTE this changes target_std[3], which rescales the
        # pressure term inside the data loss relative to velocity.
        self.pressure_gauge = pressure_gauge
        # Reflection augmentation (TRAINING ONLY -- never pass this for val/test).
        # NS is invariant under HORIZONTAL reflection, so the mirror of a valid
        # flow is the valid flow of the mirrored room. Reflect position AND the
        # velocity component along each axis; scalars (p, rho, leak magnitude)
        # are unchanged. Each entry multiplies the geometry count:
        # 'x','y','xy' => up to 4x the training data.

        # Point-cloud content fed to the geometry encoder:
        #   "full" -> xyz + 5 class one-hots + (blinded) u,v,w,p  (12 channels)
        #   "geom" -> xyz only (3 channels): the "no boundary encoder" ablation.
        if pc_mode not in ("full", "geom"):
            raise ValueError(f"pc_mode must be 'full' or 'geom', got {pc_mode!r}")
        self.pc_mode = pc_mode
        self.pc_channels = 3 if pc_mode == "geom" else 12

        self._axis = {"x": 0, "y": 1}
        self.augment = []
        for a in (augment or []):
            if a not in ("x", "y", "xy"):
                raise ValueError(f"augment must be x|y|xy (horizontal only), got {a!r}")
            self.augment.append(tuple(self._axis[c] for c in a))

        self.n_colloc = n_collocation_batch
        self.n_super = n_supervised_batch
        self.n_wall_batch = n_boundary_batch
        self.n_in_batch = n_inlet_batch
        self.n_out_batch = n_outlet_batch
        self.n_leak_batch = n_leak_batch

        self.cols_xyz = ['X (m)', 'Y (m)', 'Z (m)']
        self.cols_uvwp = ['Velocity[i] (m/s)', 'Velocity[j] (m/s)',
                          'Velocity[k] (m/s)', 'Pressure (Pa)']
        self.col_rho = 'Density (kg/m^3)'
        self.col_velmag = 'Velocity: Magnitude (m/s)'

        self.cases = []

        print("\n--- INITIALIZING NORMALIZED MULTI-CASE DATASET ---")
        self._build_dataset(n_case_pool, n_boundary_pool, saved_stats)
        print(f"--- DATASET READY ({len(self.cases)} Rooms Loaded) ---\n")

    def _build_dataset(self, n_case_pool, n_boundary_pool, saved_stats):
        if isinstance(self.case_dirs, list):
            case_folders = self.case_dirs
        else:
            case_folders = [f.path for f in os.scandir(self.case_dirs) if f.is_dir()]
            if len(case_folders) == 0:
                case_folders = [self.case_dirs]

        print(f"Found {len(case_folders)} Case Folders.")

        raw_case_data = []
        all_global_xyz, all_global_tgt = [], []
        all_rho = []
        all_interior_p = []  # A4: gauge-centered interior pressures, stats source

        # PHASE 1a: load all rooms.
        for folder in case_folders:
            raw_case_data.append(self._load_raw_room(folder))

        # PHASE 1b: reflection augmentation (training only). Each mirror is a new,
        # physically-valid room, appended alongside the originals.
        if self.augment:
            n_orig = len(raw_case_data)
            mirrored = []
            for room in raw_case_data:
                for axes in self.augment:
                    mirrored.append(self._reflect_room(room, axes))
            raw_case_data.extend(mirrored)
            print(f"Augmentation: {n_orig} rooms x {1 + len(self.augment)} "
                  f"(orig + {[''.join(k for k,v in self._axis.items() if v in ax) for ax in self.augment]}) "
                  f"= {len(raw_case_data)} rooms.")

        # PHASE 1c: accumulate normalization pools over the FULL (augmented) set.
        for room_data in raw_case_data:
            all_global_xyz.append(torch.cat([room_data['case_xyz'], room_data['wall_xyz'],
                                             room_data['in_xyz'], room_data['out_xyz'],
                                             room_data['leak_xyz']], dim=0))
            all_global_tgt.append(torch.cat([room_data['case_tgt'], room_data['wall_tgt'],
                                             room_data['in_tgt'], room_data['out_tgt'],
                                             room_data['leak_tgt']], dim=0))
            if room_data['has_rho'] and len(room_data['case_rho']) > 0:
                all_rho.append(room_data['case_rho'])
            if len(room_data['case_tgt']) > 0:
                all_interior_p.append(room_data['case_tgt'][:, 3])

        if saved_stats is None:
            print("Calculating Global Normalization Baseline...")
            mega_xyz = torch.cat(all_global_xyz, dim=0)
            mega_tgt = torch.cat(all_global_tgt, dim=0)

            self.coord_min = mega_xyz.min(dim=0)[0]
            self.coord_max = mega_xyz.max(dim=0)[0]
            self.coord_scale = self.coord_max - self.coord_min
            self.coord_scale = torch.where(self.coord_scale < 1e-6,
                                           torch.ones_like(self.coord_scale), self.coord_scale)

            self.target_mean = mega_tgt.mean(dim=0)
            self.target_std = mega_tgt.std(dim=0)
            self.target_std = torch.where(self.target_std < 1e-6,
                                          torch.ones_like(self.target_std), self.target_std)

            # A4: pressure stats from INTERIOR points only (gauge already removed
            # per case). The pooled mega_tgt would mix fake wall/leak zeros and,
            # before gauge removal, between-case offsets into the pressure std.
            if self.pressure_gauge and all_interior_p:
                interior_p = torch.cat(all_interior_p, dim=0)
                self.target_mean[3] = interior_p.mean()      # ~0 by construction
                p_std = interior_p.std()
                self.target_std[3] = p_std if p_std > 1e-6 else torch.tensor(1.0)
                print(f"[stats] pressure channel (interior-only, gauge-centered): "
                      f"mean={self.target_mean[3].item():+.4f}, "
                      f"std={self.target_std[3].item():.4f} Pa")

            # Reference density, retained for reporting.
            if all_rho:
                self.rho_ref = torch.cat(all_rho, dim=0).mean()
            else:
                self.rho_ref = torch.tensor(1.175)
                print("[WARNING] No density column found in any case: "
                      "using the default reference density.")
            print(f"[stats] rho_ref = {self.rho_ref.item():.6f} kg/m^3")

            torch.save({
                'coord_min': self.coord_min, 'coord_scale': self.coord_scale,
                'target_mean': self.target_mean, 'target_std': self.target_std,
                'rho_ref': self.rho_ref
            }, self.stats_path)
            print(f"[stats] Saved Global Normalization Baseline to '{self.stats_path}'.")
            del mega_xyz, mega_tgt
        else:
            print(f"Loading Locked Normalization Baseline from {saved_stats}...")
            stats = torch.load(saved_stats, weights_only=True)
            self.coord_min = stats['coord_min']
            self.coord_scale = stats['coord_scale']
            self.target_mean = stats['target_mean']
            self.target_std = stats['target_std']
            self.rho_ref = stats.get('rho_ref', torch.tensor(1.175))
            print(f"[stats] rho_ref = {float(self.rho_ref):.6f} kg/m^3")

        del all_global_xyz, all_global_tgt, all_rho, all_interior_p

        # PHASE 2: normalize each room individually.
        for idx, room in enumerate(raw_case_data):
            print(f"Processing Room {idx + 1}/{len(raw_case_data)}...")
            processed_room = self._process_single_room(room, n_case_pool, n_boundary_pool)
            self.cases.append(processed_room)

    def _load_raw_room(self, folder):
        all_files = glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True)
        if not all_files:
            raise FileNotFoundError(
                f"No CSV files found under '{folder}'.\n"
                f"  exists={os.path.isdir(folder)}. On Kaggle this usually means the "
                f"dataset is not attached to the notebook, or its folder name differs "
                f"from the one built by case(i). Check the left-hand Input panel.")
        dfs_case, dfs_wall, dfs_in, dfs_out, dfs_leak = [], [], [], [], []

        for f in all_files:
            filename = os.path.basename(f).lower()

            # A1: only drop rows that are missing a column we actually consume.
            df = pd.read_csv(f).replace([np.inf, -np.inf], np.nan)
            use_cols = [c for c in self.cols_xyz + self.cols_uvwp if c in df.columns]
            df = df.dropna(subset=use_cols)

            for col in self.cols_uvwp:
                if col not in df.columns:
                    df[col] = 0.0

            if "fluid_data" in filename:
                dfs_case.append(df)
            elif "leak" in filename:
                # A3: keep the magnitude column as-is; never fabricate a direction.
                dfs_leak.append(df)
            elif "new_hvac" in filename:
                for col in ['Velocity[i] (m/s)', 'Velocity[j] (m/s)', 'Velocity[k] (m/s)']:
                    df[col] = df[col].apply(lambda x: 0.0 if abs(x) < 1e-4 else x)

                vel_mag = np.sqrt(df['Velocity[i] (m/s)'] ** 2 + df['Velocity[j] (m/s)'] ** 2
                                  + df['Velocity[k] (m/s)'] ** 2)
                df_active = df[vel_mag > 0.01]
                df_dead = df[vel_mag <= 0.01]

                # A5: classify each HVAC PATCH by its net flux, not each point by
                # the sign of its own w.
                #
                # The original rule (`w < 0` -> inlet, `w > 0` -> outlet) infers a
                # BOUNDARY TYPE from the SOLUTION. Vent type is a property of the
                # setup; w is what the solver produced. The two disagree at a
                # patch's edge: a ceiling return's fringe cells sit against solid
                # ceiling (velocity -> 0, sign arbitrary) and ~1.5 m from a supply
                # jet that entrains air downward, so they read w < 0 despite
                # belonging to a return. Measured: those fringe cells are
                # -0.08..-0.13 m/s against a true supply of -1.160 m/s and a return
                # core of +0.87 m/s -- an order of magnitude weaker than real
                # supply. Consequence of the old rule: 1,919 of 14,735 inlet-class
                # points (13.0%) sat on a return patch and carried the inlet class
                # into `pc_full`, telling the geometry encoder "supply vent here"
                # exactly where the flow reverses.
                #
                # `New_HVAC.csv` has no patch/region column, so the patch must be
                # recovered geometrically: supply and return patches are compact and
                # well separated (~1.5 m apart here), so single-linkage clustering
                # gives them cleanly. Each cluster is then labelled by the sign of
                # its MEAN w -- the net flux, which is what defines a supply or a
                # return. This has no distance threshold to tune against the data
                # and is symmetric: it equally corrects a supply whose fringe reads
                # positive. Set `hvac_patch_cluster_m=0` to restore the old rule.
                if not df_active.empty and self.hvac_patch_cluster_m > 0:
                    xyz_h = df_active[self.cols_xyz].to_numpy()
                    w_h = df_active['Velocity[k] (m/s)'].to_numpy()
                    if len(xyz_h) > 1:
                        from scipy.cluster.hierarchy import fcluster, linkage
                        lab = fcluster(linkage(xyz_h, method="single"),
                                       t=self.hvac_patch_cluster_m,
                                       criterion="distance")
                    else:
                        lab = np.ones(len(xyz_h), dtype=int)
                    # A patch supplies air if its net vertical flux is downward.
                    is_supply = np.zeros(len(xyz_h), dtype=bool)
                    for k in np.unique(lab):
                        m = lab == k
                        is_supply[m] = w_h[m].mean() < 0.0
                    df_in = df_active[is_supply]
                    df_out = df_active[~is_supply]
                elif not df_active.empty:
                    df_in = df_active[df_active['Velocity[k] (m/s)'] < 0.0]
                    df_out = df_active[df_active['Velocity[k] (m/s)'] > 0.0]
                else:
                    df_in = df_out = df_active

                if not df_in.empty:
                    dfs_in.append(df_in)
                if not df_out.empty:
                    dfs_out.append(df_out)

                if not df_dead.empty:
                    df_dead_copy = df_dead.copy()
                    for col in self.cols_uvwp:
                        df_dead_copy[col] = 0.0
                    dfs_wall.append(df_dead_copy)
            else:
                for col in self.cols_uvwp:
                    df[col] = 0.0
                dfs_wall.append(df)

        dummy_cols = self.cols_xyz + self.cols_uvwp
        df_case = pd.concat(dfs_case, ignore_index=True).fillna(0.0) if dfs_case else pd.DataFrame(columns=dummy_cols)
        df_wall = pd.concat(dfs_wall, ignore_index=True).fillna(0.0) if dfs_wall else pd.DataFrame(columns=dummy_cols)
        df_in = pd.concat(dfs_in, ignore_index=True).fillna(0.0) if dfs_in else pd.DataFrame(columns=dummy_cols)
        df_out = pd.concat(dfs_out, ignore_index=True).fillna(0.0) if dfs_out else pd.DataFrame(columns=dummy_cols)
        df_leak = pd.concat(dfs_leak, ignore_index=True).fillna(0.0) if dfs_leak else pd.DataFrame(columns=dummy_cols)

        if len(df_case) == 0:
            found = sorted(os.path.basename(f) for f in all_files)
            raise ValueError(
                f"No interior points loaded for '{folder}'. A case needs a CSV whose "
                f"filename contains 'fluid_data' (case-insensitive).\n"
                f"  {len(all_files)} CSV(s) found: {found[:12]}"
                f"{' ...' if len(found) > 12 else ''}\n"
                f"  If this folder holds a nested subdirectory per case, point "
                f"case_dirs at the inner folders instead.")

        case_xyz, case_tgt = self._extract_tensors(df_case)
        wall_xyz, wall_tgt = self._extract_tensors(df_wall)
        in_xyz, in_tgt = self._extract_tensors(df_in)
        out_xyz, out_tgt = self._extract_tensors(df_out)
        leak_xyz, leak_tgt = self._extract_tensors(df_leak)

        # A4: remove this case's arbitrary pressure gauge. The interior mean is
        # the gauge; inlet/outlet come from the same solve, so they shift by the
        # same constant. Wall/leak pressures are fake zeros and are never used in
        # any pressure loss, so they are left untouched.
        if self.pressure_gauge and len(case_tgt) > 0:
            p_gauge = case_tgt[:, 3].mean()
            case_tgt[:, 3] -= p_gauge
            if len(in_tgt) > 0:
                in_tgt[:, 3] -= p_gauge
            if len(out_tgt) > 0:
                out_tgt[:, 3] -= p_gauge
            p = case_tgt[:, 3]
            print(f"  [{os.path.basename(os.path.normpath(folder))}] pressure gauge "
                  f"removed: {p_gauge.item():+.3f} Pa | centered interior p: "
                  f"std={p.std().item():.3f}, range [{p.min().item():.2f}, "
                  f"{p.max().item():.2f}] Pa")

        # A2 / A3: auxiliary scalars.
        case_rho, has_rho = self._extract_scalar(df_case, self.col_rho)
        leak_mag, _ = self._extract_scalar(df_leak, self.col_velmag)

        if not has_rho:
            print(f"[WARNING] '{self.col_rho}' missing in {os.path.basename(folder)}: "
                  "density defaults to the global reference for this case.")

        return {
            'case_xyz': case_xyz, 'case_tgt': case_tgt,
            'wall_xyz': wall_xyz, 'wall_tgt': wall_tgt,
            'in_xyz': in_xyz, 'in_tgt': in_tgt,
            'out_xyz': out_xyz, 'out_tgt': out_tgt,
            'leak_xyz': leak_xyz, 'leak_tgt': leak_tgt,
            'case_rho': case_rho, 'has_rho': has_rho, 'leak_mag': leak_mag
        }

    def _reflect_room(self, room, axes):
        """Mirror a raw room about its own bounding box on the given axes.

        `axes` is a tuple of coordinate indices to flip (0=x, 1=y; never 2=z).
        For each flipped axis a: position component a -> (min+max) - pos, and
        velocity component a -> -velocity. The velocity component index equals the
        coordinate axis index (u<->x, v<->y). Pressure (component 3), density,
        and leak magnitude are scalars/vertical and carry over unchanged. Reflecting about the room's own box keeps the
        mirror in the same extent, so global normalization is unaffected.
        """
        groups = [('case_xyz', 'case_tgt'), ('wall_xyz', 'wall_tgt'),
                  ('in_xyz', 'in_tgt'), ('out_xyz', 'out_tgt'),
                  ('leak_xyz', 'leak_tgt')]

        present = [room[g] for g, _ in groups if len(room[g]) > 0]
        all_xyz = torch.cat(present, dim=0)
        rmin, rmax = all_xyz.min(dim=0)[0], all_xyz.max(dim=0)[0]

        out = dict(room)  # carry scalars/flags (has_rho, ...)
        for g_xyz, g_tgt in groups:
            xyz, tgt = room[g_xyz].clone(), room[g_tgt].clone()
            for a in axes:
                if len(xyz) > 0:
                    xyz[:, a] = (rmin[a] + rmax[a]) - xyz[:, a]
                    tgt[:, a] = -tgt[:, a]          # velocity component a
            out[g_xyz], out[g_tgt] = xyz, tgt
        for k in ('case_rho', 'leak_mag'):
            if torch.is_tensor(room[k]):
                out[k] = room[k].clone()
        return out

    def _process_single_room(self, room, n_case_pool, n_boundary_pool):
        """Z-score normalization and point-cloud creation."""
        all_case_xyz, all_case_tgt = room['case_xyz'], room['case_tgt']
        all_wall_xyz, all_wall_tgt = room['wall_xyz'], room['wall_tgt']
        all_in_xyz, all_in_tgt = room['in_xyz'], room['in_tgt']
        all_out_xyz, all_out_tgt = room['out_xyz'], room['out_tgt']
        all_leak_xyz, all_leak_tgt = room['leak_xyz'], room['leak_tgt']

        # 1. Stratify interior points by velocity magnitude (jet stream vs background).
        u, v, w = all_case_tgt[:, 0], all_case_tgt[:, 1], all_case_tgt[:, 2]
        vel_mag = torch.sqrt(u ** 2 + v ** 2 + w ** 2)

        top_20_percent_speed = torch.quantile(vel_mag, 0.80)
        is_stream = ((vel_mag >= top_20_percent_speed) | (vel_mag >= 0.15)) & (vel_mag > 0.05)

        raw_stream_xyz, raw_stream_tgt = all_case_xyz[is_stream], all_case_tgt[is_stream]
        raw_bg_xyz, raw_bg_tgt = all_case_xyz[~is_stream], all_case_tgt[~is_stream]

        n_stream_keep = min(len(raw_stream_xyz), int(n_case_pool * 0.4))
        n_bg_keep = min(len(raw_bg_xyz), n_case_pool - n_stream_keep)

        idx_stream = torch.randperm(len(raw_stream_xyz))[:n_stream_keep]
        idx_bg = torch.randperm(len(raw_bg_xyz))[:n_bg_keep]

        pool_stream_xyz = (raw_stream_xyz[idx_stream] - self.coord_min) / self.coord_scale
        pool_stream_tgt = (raw_stream_tgt[idx_stream] - self.target_mean) / self.target_std
        pool_bg_xyz = (raw_bg_xyz[idx_bg] - self.coord_min) / self.coord_scale
        pool_bg_tgt = (raw_bg_tgt[idx_bg] - self.target_mean) / self.target_std

        wall_pool_idx = torch.randperm(len(all_wall_xyz))[:n_boundary_pool]
        pool_wall_xyz = (all_wall_xyz[wall_pool_idx] - self.coord_min) / self.coord_scale
        pool_wall_tgt = (all_wall_tgt[wall_pool_idx] - self.target_mean) / self.target_std

        pool_in_xyz = (all_in_xyz - self.coord_min) / self.coord_scale
        pool_in_tgt = (all_in_tgt - self.target_mean) / self.target_std
        pool_out_xyz = (all_out_xyz - self.coord_min) / self.coord_scale
        pool_out_tgt = (all_out_tgt - self.target_mean) / self.target_std
        pool_leak_xyz = (all_leak_xyz - self.coord_min) / self.coord_scale
        pool_leak_tgt = (all_leak_tgt - self.target_mean) / self.target_std
        # A3: leak magnitude stays in PHYSICAL m/s (it is not a per-component target).
        pool_leak_mag = room['leak_mag']
        if len(pool_leak_mag) != len(pool_leak_xyz):
            pool_leak_mag = torch.zeros(len(pool_leak_xyz), dtype=torch.float32)

        # Blinding logic: hide targets that must not leak into the point-cloud input.
        pc_wall_tgt = pool_wall_tgt.clone(); pc_wall_tgt[:, 3] = 0.0
        pc_in_tgt = pool_in_tgt.clone(); pc_in_tgt[:, 3] = 0.0
        pc_out_tgt = pool_out_tgt.clone(); pc_out_tgt[:, 3] = 0.0
        pc_leak_tgt = pool_leak_tgt.clone(); pc_leak_tgt[:, 0:3] = 0.0

        wall_mask_oh = torch.nn.functional.one_hot(torch.full((len(pool_wall_xyz),), 1), num_classes=5).float()
        in_mask_oh = torch.nn.functional.one_hot(torch.full((len(pool_in_xyz),), 2), num_classes=5).float()
        out_mask_oh = torch.nn.functional.one_hot(torch.full((len(pool_out_xyz),), 3), num_classes=5).float()
        leak_mask_oh = torch.nn.functional.one_hot(torch.full((len(pool_leak_xyz),), 0), num_classes=5).float()

        wall_feat = torch.cat([wall_mask_oh, pc_wall_tgt], dim=1)
        in_feat = torch.cat([in_mask_oh, pc_in_tgt], dim=1)
        out_feat = torch.cat([out_mask_oh, pc_out_tgt], dim=1)
        leak_feat = torch.cat([leak_mask_oh, pc_leak_tgt], dim=1)

        n_in_keep, n_out_keep = len(pool_in_xyz), len(pool_out_xyz)
        remaining_pc = max(0, self.n_pc - (n_in_keep + n_out_keep))
        n_leak_keep = min(len(pool_leak_xyz), int(remaining_pc * 0.5))
        n_wall_keep = min(len(pool_wall_xyz), remaining_pc - n_leak_keep)

        leak_pc_idx = torch.randperm(len(pool_leak_xyz))[:n_leak_keep]
        wall_pc_idx = torch.randperm(len(pool_wall_xyz))[:n_wall_keep]

        pc_xyz = torch.cat([pool_wall_xyz[wall_pc_idx], pool_in_xyz, pool_out_xyz,
                            pool_leak_xyz[leak_pc_idx]], dim=0)
        pc_feat = torch.cat([wall_feat[wall_pc_idx], in_feat, out_feat,
                             leak_feat[leak_pc_idx]], dim=0)

        shuffle_idx = torch.randperm(len(pc_xyz))
        if self.pc_mode == "geom":
            # Boundary-encoder ablation: feed ONLY surface positions (xyz), with no
            # class one-hots and no boundary flow values. The model then knows the
            # room shape but not where the vents are or what they do -- isolating
            # the value of encoding the boundary conditions.
            pc_full = pc_xyz[shuffle_idx]
        else:
            pc_full = torch.cat([pc_xyz[shuffle_idx], pc_feat[shuffle_idx]], dim=1)

        return {
            'pool_stream_xyz': pool_stream_xyz, 'pool_stream_tgt': pool_stream_tgt,
            'pool_bg_xyz': pool_bg_xyz, 'pool_bg_tgt': pool_bg_tgt,
            'pool_wall_xyz': pool_wall_xyz, 'pool_wall_tgt': pool_wall_tgt,
            'pool_in_xyz': pool_in_xyz, 'pool_in_tgt': pool_in_tgt,
            'pool_out_xyz': pool_out_xyz, 'pool_out_tgt': pool_out_tgt,
            'pool_leak_xyz': pool_leak_xyz, 'pool_leak_tgt': pool_leak_tgt,
            'pool_leak_mag': pool_leak_mag,
            'pc_full': pc_full
        }

    def _extract_tensors(self, df):
        # An empty DataFrame built from `columns=...` has object dtype, which
        # torch.tensor rejects with an opaque TypeError -- return correctly
        # shaped empty float tensors instead.
        if len(df) == 0:
            return (torch.zeros((0, len(self.cols_xyz)), dtype=torch.float32),
                    torch.zeros((0, len(self.cols_uvwp)), dtype=torch.float32))
        xyz = df[self.cols_xyz].to_numpy(dtype=np.float32)
        tgt = df[self.cols_uvwp].to_numpy(dtype=np.float32)
        return torch.from_numpy(xyz), torch.from_numpy(tgt)

    def _extract_scalar(self, df, col):
        """Return (tensor, present_flag) for an optional scalar column."""
        if col in df.columns and len(df) > 0:
            return torch.from_numpy(df[col].to_numpy(dtype=np.float32)), True
        return torch.zeros(len(df), dtype=torch.float32), col in df.columns

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        room = self.cases[idx]

        n_case_needed = self.n_colloc + self.n_super
        n_stream = int(n_case_needed * 0.60)
        n_bg = n_case_needed - n_stream

        if len(room['pool_stream_xyz']) > 0:
            idx_stream = torch.randint(high=len(room['pool_stream_xyz']), size=(n_stream,))
        else:
            idx_stream = torch.tensor([], dtype=torch.long)

        if len(room['pool_bg_xyz']) > 0:
            idx_bg = torch.randint(high=len(room['pool_bg_xyz']), size=(n_bg,))
        else:
            idx_bg = torch.tensor([], dtype=torch.long)

        batch_case_xyz = torch.cat([room['pool_stream_xyz'][idx_stream], room['pool_bg_xyz'][idx_bg]], dim=0)
        batch_case_tgt = torch.cat([room['pool_stream_tgt'][idx_stream], room['pool_bg_tgt'][idx_bg]], dim=0)
        # Track which sampled points came from the fast ("stream") pool so the
        # loss can balance stream vs background explicitly.
        batch_case_stream = torch.cat([
            torch.ones(len(idx_stream), dtype=torch.bool),
            torch.zeros(len(idx_bg), dtype=torch.bool)], dim=0)

        shuffle_idx = torch.randperm(len(batch_case_xyz))
        batch_case_xyz = batch_case_xyz[shuffle_idx]
        batch_case_tgt = batch_case_tgt[shuffle_idx]
        batch_case_stream = batch_case_stream[shuffle_idx]
        batch_case_mask = torch.full((len(batch_case_xyz),), 0, dtype=torch.long)

        n_w = min(self.n_wall_batch, len(room['pool_wall_xyz']))
        wall_idx = torch.randperm(len(room['pool_wall_xyz']))[:n_w]
        batch_wall_xyz, batch_wall_tgt = room['pool_wall_xyz'][wall_idx], room['pool_wall_tgt'][wall_idx]
        batch_wall_mask = torch.full((len(batch_wall_xyz),), 1, dtype=torch.long)

        n_i = min(self.n_in_batch, len(room['pool_in_xyz']))
        in_idx = torch.randperm(len(room['pool_in_xyz']))[:n_i]
        batch_in_xyz, batch_in_tgt = room['pool_in_xyz'][in_idx], room['pool_in_tgt'][in_idx]
        batch_in_mask = torch.full((len(batch_in_xyz),), 2, dtype=torch.long)

        n_o = min(self.n_out_batch, len(room['pool_out_xyz']))
        out_idx = torch.randperm(len(room['pool_out_xyz']))[:n_o]
        batch_out_xyz, batch_out_tgt = room['pool_out_xyz'][out_idx], room['pool_out_tgt'][out_idx]
        batch_out_mask = torch.full((len(batch_out_xyz),), 3, dtype=torch.long)

        n_l = min(self.n_leak_batch, len(room['pool_leak_xyz']))
        leak_idx = torch.randperm(len(room['pool_leak_xyz']))[:n_l]
        batch_leak_xyz, batch_leak_tgt = room['pool_leak_xyz'][leak_idx], room['pool_leak_tgt'][leak_idx]
        batch_leak_mag = room['pool_leak_mag'][leak_idx]
        batch_leak_mask = torch.full((len(batch_leak_xyz),), 4, dtype=torch.long)

        batch_xyt = torch.cat([batch_case_xyz, batch_wall_xyz, batch_in_xyz,
                               batch_out_xyz, batch_leak_xyz], dim=0)
        batch_targets = torch.cat([batch_case_tgt, batch_wall_tgt, batch_in_tgt,
                                   batch_out_tgt, batch_leak_tgt], dim=0)
        batch_masks = torch.cat([batch_case_mask, batch_wall_mask, batch_in_mask,
                                 batch_out_mask, batch_leak_mask], dim=0)

        total_case_points = len(room['pool_stream_xyz']) + len(room['pool_bg_xyz'])

        return {
            "pc": room['pc_full'],
            "xyt": batch_xyt,
            "targets": batch_targets,
            "bc_mask": batch_masks,
            # Aligned with the leading case-point block of `xyt` (collocation first).
            "is_stream": batch_case_stream,
            "leak_mag": batch_leak_mag,
            "n_colloc": torch.tensor([min(self.n_colloc, total_case_points)]),
            "n_super": torch.tensor([min(self.n_super, max(0, total_case_points - self.n_colloc))]),
            "n_boundary_total": torch.tensor([len(batch_wall_xyz) + len(batch_in_xyz)
                                              + len(batch_out_xyz) + len(batch_leak_xyz)]),
            "coord_scale": self.coord_scale,
            "target_mean": self.target_mean,
            "target_std": self.target_std
        }
