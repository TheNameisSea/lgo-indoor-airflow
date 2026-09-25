"""Zone-aware checkpoint selection.

WHY. `ThermoTrainer.validate()` ends with `val_combined = -score`, and the base
trainer checkpoints on minimising that key -- so one line is the whole selection
policy. `score` is the joint Taylor score over ALL interior points, and it is
variance-weighted: the near-jet zone holds 94% of the field variance in 29% of
the points, so the criterion is effectively a near-jet score.

Across the plateau where the global score is flat, occupied-zone skill still
varies, so the global criterion picks among near-equal epochs almost at random
with respect to the zones that matter for comfort.

THE CRITERION. `zone_balanced` is the mean of the three zones' joint scores --
equal weight per REGION rather than per variance. Near-jet is already at 100% of
the R2 its correlation allows, so it has little to gain and much to lose; giving
the occupied zone a third of the vote instead of ~6% is the point. `occupied`
selects on that zone alone and is offered for the ablation, not as a default.

COST. Nothing measurable: validate() already predicts 50k points per case, and
this only buckets them. The zone label of each point is computed once and cached,
which is exact because validate() re-seeds its sampler to 0 every epoch and
therefore scores the same points each time.

CAVEAT. This is model selection on the validation set. So is the existing
criterion, so it is not a new kind of leakage -- but a narrower criterion raises
the risk of fitting the 7 val cases, and the test split stays the only clean
measurement.
"""

import numpy as np
import torch

from thermo.metrics_joint import joint_metrics, summarise
from thermo.trainer_thermo import ThermoTrainer

ZONES = (("near_jet", 0.0, 1.0), ("transition", 1.0, 2.0), ("occupied", 2.0, np.inf))
# ASHRAE 55 occupied zone, reproduced from the room dict alone. The trainer only
# has the POOLED boundary cloud (walls and furniture merged), so it cannot class
# surfaces by filename the way evaluate.py does -- but it does not need to: the
# exterior walls ARE the room's bounding box, so a 1.0 m inset in x/y is that
# rule exactly, and the internal walls (a column at x <= 0.2 m, a board at
# y = 0.3 m) fall inside the inset anyway. A small setback from the pooled solids
# then removes the boundary layer on furniture. Validated against evaluate.py's
# exact per-class mask on Case_07: IoU 0.983.
ASHRAE_BANDS = (("ashrae_seated", 0.10, 1.35), ("ashrae_standing", 0.10, 1.80))
ASHRAE_INSET_M = 1.0        # from the exterior walls (the room bbox in x, y)
ASHRAE_FURNITURE_M = 0.1    # from any pooled solid: boundary layer only

CRITERIA = ("joint", "zone_balanced", "occupied", "ashrae", "comfort_balanced")

# The ASHRAE band is ~2.9% of the interior, and ThermoTrainer.validate() samples
# 50k points per case UNIFORMLY -- so the band receives about 1,464 points per
# case, 10,245 over the 7 val cases. Steering checkpoint selection on that is
# steering on sampling noise.
#
# The fix is not a wider band -- widening it past 1.80 m stops being ASHRAE, and
# shrinking the setbacks puts the boundary layer back in, which is what made the
# distance-band metric misleading in the first place. The fix is to sample the
# band DIRECTLY: 20k band points per case is 3.7x the points at 0.27x the noise,
# where reaching only 0.35x by raising uniform sampling would cost 8x the
# validation compute.
N_BAND_POINTS = 20_000


class ZoneThermoTrainer(ThermoTrainer):
    """ThermoTrainer whose checkpoint criterion can weight regions equally."""

    def __init__(self, *a, select_on="zone_balanced", n_band=N_BAND_POINTS, **kw):
        super().__init__(*a, **kw)
        self.n_band = int(n_band)
        self._band_cache = {}
        if select_on not in CRITERIA:
            raise ValueError(f"select_on must be one of {CRITERIA}, got {select_on!r}")
        self.select_on = select_on
        self._zone_cache = {}

    def _zone_labels(self, room, xyz_norm, ds_scale, ds_min, key):
        """Distance-to-nearest-supply-vent band for each sampled point.

        Cached: validate() re-seeds its sampler to 0 each epoch, so the sampled
        points never change and this is computed once per case.
        """
        if key in self._zone_cache:
            return self._zone_cache[key]
        from scipy.spatial import cKDTree
        sup = room.get("pool_in_xyz")
        if sup is None or len(sup) == 0:
            self._zone_cache[key] = None
            return None
        s = sup.numpy().astype(np.float64) * ds_scale + ds_min
        q = xyz_norm.numpy().astype(np.float64) * ds_scale + ds_min
        d, _ = cKDTree(s).query(q, k=1, workers=-1)
        masks = {n: (d >= lo) & (d < hi) for n, lo, hi in ZONES}

        # ASHRAE bands, from the same sampled points.
        wall = room.get("pool_wall_xyz")
        if wall is not None and len(wall):
            w = wall.numpy().astype(np.float64) * ds_scale + ds_min
            lo_b, hi_b = w.min(0), w.max(0)
            inset = ((q[:, 0] >= lo_b[0] + ASHRAE_INSET_M)
                     & (q[:, 0] <= hi_b[0] - ASHRAE_INSET_M)
                     & (q[:, 1] >= lo_b[1] + ASHRAE_INSET_M)
                     & (q[:, 1] <= hi_b[1] - ASHRAE_INSET_M))
            dw, _ = cKDTree(w).query(q, k=1, workers=-1)
            free = inset & (dw >= ASHRAE_FURNITURE_M)
            for n, z0, z1 in ASHRAE_BANDS:
                masks[n] = free & (q[:, 2] >= z0) & (q[:, 2] <= z1)
        self._zone_cache[key] = masks
        return self._zone_cache[key]

    def _band_points(self, room, ds_scale, ds_min, key):
        """(xyz, tgt) sampled from inside the ASHRAE band, over the FULL pool.

        Fixed seed, so the same points are scored every epoch and the criterion
        moves only because the model moved. Cached: the mask costs one KD-tree
        query per case and never changes.
        """
        if key in self._band_cache:
            return self._band_cache[key]
        from scipy.spatial import cKDTree
        xyz = torch.cat([room["pool_stream_xyz"], room["pool_bg_xyz"]], dim=0)
        tgt = torch.cat([room["pool_stream_tgt"], room["pool_bg_tgt"]], dim=0)
        wall = room.get("pool_wall_xyz")
        if wall is None or not len(wall) or not len(xyz):
            self._band_cache[key] = None
            return None
        q = xyz.numpy().astype(np.float64) * ds_scale + ds_min
        w = wall.numpy().astype(np.float64) * ds_scale + ds_min
        lo, hi = w.min(0), w.max(0)
        inset = ((q[:, 0] >= lo[0] + ASHRAE_INSET_M) & (q[:, 0] <= hi[0] - ASHRAE_INSET_M)
                 & (q[:, 1] >= lo[1] + ASHRAE_INSET_M) & (q[:, 1] <= hi[1] - ASHRAE_INSET_M))
        dw, _ = cKDTree(w).query(q, k=1, workers=-1)
        free = inset & (dw >= ASHRAE_FURNITURE_M)
        out = {}
        for n, z0, z1 in ASHRAE_BANDS:
            m = np.where(free & (q[:, 2] >= z0) & (q[:, 2] <= z1))[0]
            if len(m) < 100:
                continue
            if len(m) > self.n_band:
                g = np.random.default_rng(0)
                m = m[g.choice(len(m), self.n_band, replace=False)]
            out[n] = (xyz[m], tgt[m])
        self._band_cache[key] = out or None
        return self._band_cache[key]

    @torch.no_grad()
    def validate(self, val_dataset, n_points=50000, chunk=20000):
        m = super().validate(val_dataset, n_points=n_points, chunk=chunk)
        if self.select_on == "joint":
            return m                      # unchanged behaviour, byte for byte

        tmean = val_dataset.target_mean.to(self.device)
        tstd = val_dataset.target_std.to(self.device)
        scale = val_dataset.coord_scale.numpy().astype(np.float64)
        cmin = val_dataset.coord_min.numpy().astype(np.float64)
        gen = torch.Generator().manual_seed(0)     # SAME points as super()

        was_training = self.model.training
        self.model.eval()
        per_zone = {n: [] for n, _, _ in tuple(ZONES) + tuple(ASHRAE_BANDS)}
        for ci, room in enumerate(val_dataset.cases):
            xyz = torch.cat([room["pool_stream_xyz"], room["pool_bg_xyz"]], dim=0)
            tgt = torch.cat([room["pool_stream_tgt"], room["pool_bg_tgt"]], dim=0)
            if len(xyz) == 0:
                continue
            latent = self.model.encode_geometry(room["pc_full"].unsqueeze(0).to(self.device))
            if len(xyz) > n_points:
                sel = torch.randperm(len(xyz), generator=gen)[:n_points]
                xyz, tgt = xyz[sel], tgt[sel]
            zones = self._zone_labels(room, xyz, scale, cmin, ci)
            if zones is None:
                continue
            zones = {k: v for k, v in zones.items()
                     if not k.startswith("ashrae")}      # scored separately, below
            pred = torch.cat([self.model.decode_query(
                latent, xyz[i:i + chunk].unsqueeze(0).to(self.device)).squeeze(0)
                for i in range(0, len(xyz), chunk)], dim=0)
            nch = pred.shape[-1]
            t = (tgt.to(self.device) * tstd + tmean).cpu().numpy()
            p = (pred * tstd[:nch] + tmean[:nch]).cpu().numpy()
            for n, mask in zones.items():
                if mask.sum() >= 100:
                    per_zone[n].append(joint_metrics(t[mask], p[mask]))

            # ASHRAE bands: their own, much denser sample. ~20k points per case
            # inside a band that a uniform 50k draw would hit only ~1.5k times.
            bands = self._band_points(room, scale, cmin, ci)
            for n, (bxyz, btgt) in (bands or {}).items():
                bp = torch.cat([self.model.decode_query(
                    latent, bxyz[i:i + chunk].unsqueeze(0).to(self.device)).squeeze(0)
                    for i in range(0, len(bxyz), chunk)], dim=0)
                nb = bp.shape[-1]
                per_zone[n].append(joint_metrics(
                    (btgt.to(self.device) * tstd + tmean).cpu().numpy(),
                    (bp * tstd[:nb] + tmean[:nb]).cpu().numpy()))
        if was_training:
            self.model.train()

        scores, detail = {}, {}
        for n, lst in per_zone.items():
            if lst:
                z = summarise(lst)
                scores[n] = z["score"]
                detail[n] = z
                # Recorded in history.json so a run can be re-read after the
                # fact, not just watched live.
                m[f"{n}_score"] = z["score"]
                m[f"{n}_vel_rho"] = z["vel_rho"]
                m[f"{n}_vel_r2"] = z["vel_r2"]
                m[f"{n}_vel_s"] = z["vel_s"]
                m[f"{n}_temp_mae"] = z["temp_mae"]
                m[f"{n}_n"] = int(sum(c.get("n_points", 0) for c in lst)) or len(lst)
        if not scores:
            return m

        dist_zones = [scores[n] for n, _, _ in ZONES if n in scores]
        ashrae = [scores[n] for n, _, _ in ASHRAE_BANDS if n in scores]
        m["zone_balanced"] = float(np.mean(dist_zones)) if dist_zones else m["score"]

        # Print the zones every validation. The base trainer's log line reads a
        # fixed set of keys and cannot show these, and the ASHRAE band is the
        # metric the project is actually judged on -- watching it live is how a
        # bad run gets caught early instead of three hours later.
        for n in [z[0] for z in ZONES] + [b[0] for b in ASHRAE_BANDS]:
            z = detail.get(n)
            if z is None:
                continue
            print(f"    [{n:15}] rho={z['vel_rho']:+.3f} R2={z['vel_r2']:+.3f} "
                  f"s={z['vel_s']:.2f} T_mae={z['temp_mae']:.3f}K "
                  f"S={z['score']:.4f}", flush=True)
        if ashrae:
            m["ashrae_score"] = float(np.mean(ashrae))

        if self.select_on == "zone_balanced":
            sel_score = m["zone_balanced"]
        elif self.select_on == "occupied":
            sel_score = scores.get("occupied", m["score"])
        elif self.select_on == "ashrae":
            sel_score = m.get("ashrae_score", m["score"])
        else:                                   # comfort_balanced
            # Half the vote to the near jet, half to the comfort zone. Pure
            # ASHRAE selection is tempting but the band is ~3% of points, so it
            # is the noisiest thing we could steer on, and nothing would then
            # defend the jets -- which carry 94% of the variance and are already
            # at 100% of the R2 their correlation allows, i.e. all to lose.
            near = scores.get("near_jet", m["score"])
            comf = m.get("ashrae_score", scores.get("occupied", m["score"]))
            sel_score = 0.5 * (near + comf)
        m["select_score"] = sel_score
        m["val_combined"] = -sel_score    # minimise-key => maximise the criterion
        print(f"    [select: {self.select_on}] {sel_score:.4f}"
              f"   (global joint score {m['score']:.4f})", flush=True)
        return m

    def train(self, *a, **kw):
        print(f"[zone] checkpoint criterion: {self.select_on} "
              f"(zones {', '.join(n for n, _, _ in ZONES)})", flush=True)
        return super().train(*a, **kw)
