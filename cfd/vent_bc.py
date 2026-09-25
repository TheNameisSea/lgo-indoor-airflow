#!/usr/bin/env python
"""Emit OpenFOAM boundary conditions for the vents from the fitted profile.

`step_to_stl.py` replaced each diffuser with a flat patch in the ceiling, and
`fit_vent_profile.py` measured what the deleted vanes were doing to the air.
This turns that profile into `0/U` and `0/T` entries.

Form of the supply condition, per face, with r measured from the panel centre:

    U = |w_supply| * ( u_r(r) * rhat  +  u_z(r) * zhat )
    T = T_supply + dT(r)   where the profile flows IN
        zero-gradient      where it flows OUT

The second line is not a detail. 19% of the opening carries room air *upward*
into the deleted plenum, so a `fixedValue` temperature would inject supply-cold
air where the real flow is warm return air. `codedMixed` gives Dirichlet on the
inflow annulus and zero-gradient on the recirculating core, which is
`inletOutlet` with a per-face inlet value.

Because the profile is dimensionless, the only per-case numbers are the panel
centre and the case's own supply velocity and temperature. Moving an FCU means
changing cx/cy -- no remeshing of vane geometry, no per-case data files.

Returns are pressure outlets and the door leak is a fixed flux, both per
`case_spec.json`; they are emitted too so `0/` is complete.

Run:
    python cfd/vent_bc.py --out cfd/bc            # uses geometry/patches.json
"""

import argparse
import json
import os
import sys

TEMPLATE_U = """    {patch}
    {{
        type            codedFixedValue;
        value           uniform (0 0 0);
        name            supplyProfile_{patch};
        codeInclude
        #{{
            #include "fvCFD.H"
        #}};
        code
        #{{
            const scalar wSup = {wsup:.6f};      // |supply velocity|, m/s
            const scalar cx = {cx:.6f}, cy = {cy:.6f};
            static const label nT = {n};
            static const scalar rT[] = {{{rtab}}};
            static const scalar uzT[] = {{{uztab}}};
            static const scalar urT[] = {{{urtab}}};

            const vectorField& Cf = patch().Cf();
            vectorField out(Cf.size(), vector::zero);
            forAll(Cf, i)
            {{
                const scalar dx = Cf[i].x() - cx;
                const scalar dy = Cf[i].y() - cy;
                const scalar r  = Foam::sqrt(dx*dx + dy*dy);

                label k = 0;
                while (k < nT-2 && rT[k+1] < r) k++;
                const scalar t =
                    (r - rT[k])/max(rT[k+1] - rT[k], SMALL);
                const scalar s = min(max(t, 0.0), 1.0);
                const scalar uz = uzT[k] + s*(uzT[k+1] - uzT[k]);
                const scalar ur = urT[k] + s*(urT[k+1] - urT[k]);

                // rhat is undefined at the centre; the profile is ~0 there.
                const scalar inv = (r > SMALL) ? 1.0/r : 0.0;
                out[i] = wSup*vector(ur*dx*inv, ur*dy*inv, uz);
            }}
            operator==(out);
        #}};
    }}
"""

TEMPLATE_T = """    {patch}
    {{
        type            codedMixed;
        refValue        uniform {tsup:.4f};
        refGradient     uniform 0;
        valueFraction   uniform 1;
        value           uniform {tsup:.4f};
        name            supplyTemp_{patch};
        code
        #{{
            const scalar tSup = {tsup:.6f};
            const scalar cx = {cx:.6f}, cy = {cy:.6f};
            static const label nT = {n};
            static const scalar rT[] = {{{rtab}}};
            static const scalar uzT[] = {{{uztab}}};
            static const scalar dTT[] = {{{dttab}}};

            const vectorField& Cf = patch().Cf();
            forAll(Cf, i)
            {{
                const scalar dx = Cf[i].x() - cx;
                const scalar dy = Cf[i].y() - cy;
                const scalar r  = Foam::sqrt(dx*dx + dy*dy);

                label k = 0;
                while (k < nT-2 && rT[k+1] < r) k++;
                const scalar t =
                    (r - rT[k])/max(rT[k+1] - rT[k], SMALL);
                const scalar s = min(max(t, 0.0), 1.0);
                const scalar uz = uzT[k] + s*(uzT[k+1] - uzT[k]);
                const scalar dT = dTT[k] + s*(dTT[k+1] - dTT[k]);

                this->refValue()[i] = tSup + dT;
                this->refGrad()[i]  = 0.0;
                // Dirichlet where the profile blows in, zero-gradient where the
                // core recirculates room air back up through the opening.
                this->valueFraction()[i] = (uz < 0.0) ? 1.0 : 0.0;
            }}
        #}};
    }}
"""

TEMPLATE_RETURN_U = """    {patch}
    {{
        type            pressureInletOutletVelocity;
        value           uniform (0 0 0);
    }}
"""

TEMPLATE_RETURN_T = """    {patch}
    {{
        type            inletOutlet;
        inletValue      uniform {tsup:.4f};
        value           uniform {tsup:.4f};
    }}
"""


def fmt(values, width=6):
    return ", ".join(f"{v:.{width}f}" for v in values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="cfd/vent_profile.json")
    ap.add_argument("--patches", default="cfd/geometry/patches.json")
    ap.add_argument("--out", default="cfd/bc")
    ap.add_argument("--w-supply", type=float, default=0.5803,
                    help="|supply velocity| for this case [m/s]")
    ap.add_argument("--t-supply", type=float, default=286.15,
                    help="supply temperature for this case [K]")
    args = ap.parse_args()

    prof = json.load(open(args.profile))
    patches = json.load(open(args.patches))["patches"]
    os.makedirs(args.out, exist_ok=True)

    tab = dict(n=len(prof["r"]), rtab=fmt(prof["r"]),
               uztab=fmt(prof["u_z"]), urtab=fmt(prof["u_r"]),
               dttab=fmt(prof["dT"]))

    u_entries, t_entries = [], []
    for name, p in sorted(patches.items()):
        (lo, hi) = p["bbox_m"]
        cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
        if p["role"] == "supply":
            u_entries.append(TEMPLATE_U.format(patch=name, wsup=args.w_supply,
                                               cx=cx, cy=cy, **tab))
            t_entries.append(TEMPLATE_T.format(patch=name, tsup=args.t_supply,
                                               cx=cx, cy=cy, **tab))
        elif p["role"] == "return":
            u_entries.append(TEMPLATE_RETURN_U.format(patch=name))
            t_entries.append(TEMPLATE_RETURN_T.format(patch=name,
                                                      tsup=args.t_supply))

    with open(os.path.join(args.out, "U.vents"), "w") as f:
        f.write("".join(u_entries))
    with open(os.path.join(args.out, "T.vents"), "w") as f:
        f.write("".join(t_entries))

    sup = [n for n, p in patches.items() if p["role"] == "supply"]
    ret = [n for n, p in patches.items() if p["role"] == "return"]
    print(f"profile: {prof['panels']} panels, {prof['points']} points, "
          f"mass_scale {prof['mass_scale']:.4f}")
    print(f"supply patches ({len(sup)}): {sorted(sup)}")
    print(f"return patches ({len(ret)}): {sorted(ret)}")
    print(f"w_supply {args.w_supply} m/s, T_supply {args.t_supply} K")
    print(f"wrote {args.out}/U.vents and {args.out}/T.vents")

    # Report the flow the BC will actually deliver, as a check on the emitted
    # numbers rather than on the fit.
    import numpy as np
    a = prof["opening_m"] / 2
    g = (np.arange(400) + 0.5) / 400 * 2 * a - a
    gx, gy = np.meshgrid(g, g)
    r = np.hypot(gx, gy)
    uz = np.interp(r, prof["r"], prof["u_z"]) * args.w_supply
    q = -uz.mean() * prof["opening_m"] ** 2
    print(f"\nper supply patch: net {q:.4f} m^3/s "
          f"(target {args.w_supply*prof['panel_m']**2:.4f}), "
          f"{100*(uz>0).mean():.0f}% of the area flows back up")
    print(f"three supplies: {3*q:.4f} m^3/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
