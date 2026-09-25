#!/usr/bin/env python
"""Build a complete buoyantSimpleFoam case for one FCU layout.

Everything except vent position is frozen (`case_spec.json`), so a case is
defined by the six vent centres. Written in two stages because snappyHexMesh
decides the final patch names: `--stage mesh` writes `constant/` + `system/`,
and `--stage fields` writes `0/` after reading the names the mesher actually
produced. `--stage all` does both with the meshing in between.

Physics choices, and why:

  * **Boussinesq**, rho0 = 1.2058 kg/m3 at T0 = 292.77 K, beta = 1/T0. The
    dataset's own density column is 1.2058 mean and matches ideal gas at the
    mean temperature to 4 decimal places, so this is measured, not assumed.
    NOTE the `rho0 = 0.997` in `case_spec.json` is the reference the STAR-CCM+
    *pressure* was reduced against -- it is not an air density (it corresponds
    to 354 K) and using it here would put ~20% error into the buoyancy term.
  * **kEpsilon** with wall functions: the reference `comfortHotRoom` tutorial
    for this solver uses it, and we have no data to justify anything fancier.
  * **The leak carries its MEASURED flux**, 0.0009 m3/s (Leak.csv: 0.0678 m/s
    over 0.01331 m2), which is 0.15% of supply. This BC has been wrong twice:
    prescribing 0.232 m3/s from case_spec's supply-minus-return imbalance was
    258x too much, and making it a pressure outlet was still ~34x too much
    (2.3 m/s through the slot). The pressure-outlet version fails because it
    assumes the leak and the returns sit at the same pressure -- the returns
    are fan-driven, the undercut opens onto a corridor at room pressure -- so
    a shared p_rgh hands the leak a share set by conductance alone.

Run:
    conda activate cfdEnv
    python cfd/make_case.py --out cfd/run/base --stage all
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np

HDR = """FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    {loc}object      {obj};
}}
"""


def head(cls, obj, location=None):
    loc = f'location    "{location}";\n    ' if location else ""
    return HDR.format(cls=cls, obj=obj, loc=loc)


# --- constant/ ---------------------------------------------------------------

G = head("uniformDimensionedVectorField", "g", "constant") + """
dimensions      [0 1 -2 0 0 0 0];
value           (0 0 -9.81);
"""

THERMO = head("dictionary", "thermophysicalProperties", "constant") + """
thermoType
{
    type            heRhoThermo;
    mixture         pureMixture;
    transport       const;
    thermo          hConst;
    equationOfState Boussinesq;
    specie          specie;
    energy          sensibleEnthalpy;
}

mixture
{
    specie          { molWeight 28.96; }
    equationOfState { rho0 1.2058; T0 292.77; beta 3.4157e-03; }
    thermodynamics  { Cp 1005; Hf 0; }
    transport       { mu 1.82e-05; Pr 0.71; }
}
"""

TURB = head("dictionary", "turbulenceProperties", "constant") + """
simulationType  RAS;
RAS
{
    model           kEpsilon;
    turbulence      on;
    printCoeffs     on;
}
"""

# --- system/ -----------------------------------------------------------------

CONTROL_BODY = """
application     buoyantSimpleFoam;
startFrom       latestTime;
startTime       0;
stopAt          endTime;
endTime         {end};
deltaT          1;
// purgeWrite MUST stay 0. It applies to any utility that writes a time
// directory, not just the solver: running `postProcess -func writeCellCentres`
// writes C at time 0, which counts as a write and purged every saved solution
// including the converged one.
writeControl    timeStep;
writeInterval   {wi};
purgeWrite      0;
writeFormat     ascii;
writePrecision  7;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
"""

FVSCHEMES = head("dictionary", "fvSchemes", "system") + """
ddtSchemes      { default steadyState; }
gradSchemes     { default Gauss linear; }

divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss upwind;
    div(phi,h)      bounded Gauss upwind;
    div(phi,e)      bounded Gauss limitedLinear 1;
    div(phi,K)      bounded Gauss limitedLinear 1;
    div(phi,k)      bounded Gauss upwind;
    div(phi,epsilon) bounded Gauss upwind;
    div(phi,Ekp)    bounded Gauss limitedLinear 1;
    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes   { default corrected; }
"""

FVSOLUTION = head("dictionary", "fvSolution", "system") + """
solvers
{
    p_rgh
    {
        solver          GAMG;
        tolerance       1e-08;
        relTol          0.01;
        smoother        GaussSeidel;
    }

    "(U|h|e|k|epsilon)"
    {
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-08;
        relTol          0.1;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 0;
    pRefCell        0;
    pRefValue       0;
    residualControl
    {
        p_rgh           1e-4;
        U               1e-4;
        h               1e-4;
        "(k|epsilon)"   1e-4;
    }
}

relaxationFactors
{
    // 0.10 throughout, arrived at by measurement over three settings. The
    // criterion is a CLEAN TAIL -- no k/epsilon bounding in the final quarter,
    // i.e. the converged state is not being clipped -- not the lowest residual.
    //
    //   relax  iters        events   Q1-4              p_rgh
    //   0.20   4500            165   [.., 59 in Q4]    6.4e-03
    //   0.15   3000             44   [0, 0, 1, 43]     7.1e-03
    //   0.15   3000->6000      190   [98, 22, 33, 37]  4.0e-03
    //   0.10   3000              8   [0, 7, 1, 0]      1.2e-02   <-- adopted
    //   0.10   3000->6000       40   [0, 8, 14, 18]    5.0e-03
    //   0.10   3000, layout B    0   [0, 0, 0, 0]      1.3e-02
    //
    // Two things that table settles. MORE ITERATIONS DOES NOT HELP: at either
    // relaxation, extending to 6000 halves the residual and makes the tail
    // WORSE. The flow is mildly unsteady, so driving the solver harder toward a
    // steady state it does not quite possess is exactly what forces the
    // clipping. And the answer does not depend on the setting -- every pair
    // above agrees at corr 0.97-0.98 on U and T with no component changing
    // sign -- so this is a choice between numerics, not between physics.
    //
    // Consequence recorded in the paper: 0.10/3000 is NOT tightly
    // converged in the residual sense, and the field still moves ~8% if pushed
    // to 6000. A consistent artifact-free state is what the dataset needs.
    fields  { rho 1.0; p_rgh {relax}; }
    equations { U {relax}; "(h|e)" {relax}; "(k|epsilon)" {relax}; }
}
"""

DECOMPOSE_BODY = """
numberOfSubdomains {n};
method          scotch;
"""

MESHQUALITY = head("dictionary", "meshQualityDict", "system") + """
#includeEtc "caseDicts/meshQualityDict"
"""


FVOPTIONS = head("dictionary", "fvOptions", "system") + """
// Guard rails, not physics: the momentum-method supply patch carries a large
// tangential velocity, and an early transient can push T or U somewhere the
// thermo model cannot evaluate. Steady state should sit far inside these.
limitT
{
    type            limitTemperature;
    active          yes;
    selectionMode   all;
    min             270;
    max             340;
}

limitU
{
    type            limitVelocity;
    active          yes;
    selectionMode   all;
    max             10;
}
"""


def block_mesh_dict(lo, hi, cell):
    n = [max(1, int(round((hi[i] - lo[i]) / cell))) for i in range(3)]
    v = [(lo[0], lo[1], lo[2]), (hi[0], lo[1], lo[2]), (hi[0], hi[1], lo[2]),
         (lo[0], hi[1], lo[2]), (lo[0], lo[1], hi[2]), (hi[0], lo[1], hi[2]),
         (hi[0], hi[1], hi[2]), (lo[0], hi[1], hi[2])]
    verts = "\n".join(f"    ({a} {b} {c})" for a, b, c in v)
    return head("dictionary", "blockMeshDict", "system") + f"""
scale   1;

vertices
(
{verts}
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({n[0]} {n[1]} {n[2]}) simpleGrading (1 1 1)
);

edges ();

boundary
(
    background
    {{
        type patch;
        faces
        (
            (0 3 2 1) (4 5 6 7) (0 1 5 4)
            (2 3 7 6) (0 4 7 3) (1 2 6 5)
        );
    }}
);

mergePatchPairs ();
"""


def leak_box(patches, pad=0.02):
    """Tight box around the door undercut, for volume refinement.

    Refining the Leak *surface* is not enough: the channel behind it is bounded
    by Floor and Wall_Door faces carrying the room's own (coarse) level, so no
    cell inside the 10 mm gap is ever refined and the whole passage collapses --
    snappyHexMesh then drops the patch without an error. The box has to be kept
    tight: the same level over a loose box costs millions of cells, over this
    one a few hundred thousand.
    """
    lo, hi = (np.array(v, float) for v in patches["Leak"]["bbox_m"])
    lo, hi = lo - pad, hi + pad
    # The Leak is a zero-thickness face, so padding alone covers the face and
    # not the passage behind it. Extend the degenerate axis toward the room.
    room = np.array([np.mean(patches["Floor"]["bbox_m"], axis=0)]).ravel()
    flat = np.argmin(hi - lo)
    depth = 0.15
    if room[flat] > (lo[flat] + hi[flat]) / 2:
        hi[flat] += depth
    else:
        lo[flat] -= depth
    return lo.tolist(), hi.tolist()


def snappy_dict(patches, inside, base_level, vent_level, leak_level):
    regions = "\n".join(f"            {n} {{ name {n}; }}" for n in patches)
    vents = [n for n, p in patches.items() if p["role"] in ("supply", "return")]
    ref = []
    for n, p in sorted(patches.items()):
        # The leak is a 10 mm slot: at the surface level the rest of the room
        # uses, it is thinner than a cell and snappyHexMesh drops the patch
        # entirely -- silently, leaving the case with no path for the 0.232
        # m3/s that must leave through the door.
        lvl = {"supply": vent_level, "return": vent_level,
               "leak": leak_level}.get(p["role"], base_level)
        ptype = "patch" if p["role"] in ("supply", "return", "leak") else "wall"
        ref.append(f"                {n}\n                {{\n"
                   f"                    level ({lvl} {lvl});\n"
                   f"                    patchInfo {{ type {ptype}; }}\n"
                   f"                }}")
    ref = "\n".join(ref)
    blo, bhi = leak_box(patches)
    return head("dictionary", "snappyHexMeshDict", "system") + f"""
castellatedMesh true;
snap            true;
addLayers       false;

geometry
{{
    room.stl
    {{
        type triSurfaceMesh;
        name room;
        regions
        {{
{regions}
        }}
    }}

    leakBox
    {{
        type searchableBox;
        min ({blo[0]:.4f} {blo[1]:.4f} {blo[2]:.4f});
        max ({bhi[0]:.4f} {bhi[1]:.4f} {bhi[2]:.4f});
    }}
}}

castellatedMeshControls
{{
    maxLocalCells       2000000;
    maxGlobalCells      8000000;
    minRefinementCells  10;
    maxLoadUnbalance    0.10;
    nCellsBetweenLevels 2;

    features ();

    refinementSurfaces
    {{
        room
        {{
            level ({base_level} {base_level});
            patchInfo {{ type wall; }}
            regions
            {{
{ref}
            }}
        }}
    }}

    resolveFeatureAngle     30;
    refinementRegions
    {{
        leakBox {{ mode inside; levels ((1e15 {leak_level})); }}
    }}
    locationInMesh          ({inside[0]} {inside[1]} {inside[2]});
    allowFreeStandingZoneFaces true;
}}

snapControls
{{
    nSmoothPatch        3;
    tolerance           2.0;
    nSolveIter          50;
    nRelaxIter          5;
    nFeatureSnapIter    10;
    implicitFeatureSnap false;
    explicitFeatureSnap true;
    multiRegionFeatureSnap false;
}}

addLayersControls
{{
    relativeSizes       true;
    layers {{}};
    expansionRatio      1.2;
    finalLayerThickness 0.5;
    minThickness        0.1;
    nGrow               0;
    featureAngle        60;
    nRelaxIter          3;
    nSmoothSurfaceNormals 1;
    nSmoothNormals      3;
    nSmoothThickness    10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle  90;
    nBufferCellsNoExtrude 0;
    nLayerIter          50;
}}

meshQualityControls
{{
    #includeEtc "caseDicts/meshQualityDict"
    nSmoothScale    4;
    errorReduction  0.75;
}}

writeFlags  ( scalarLevels layerSets layerFields );
mergeTolerance 1e-6;
"""


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("/*--------------------------------*- C++ -*------------------"
                "----------------*\\\n" + "\\*----------------------------------"
                "-----------------------------------------*/\n" + text)


def stage_mesh(args, patches):
    case = args.out
    write(os.path.join(case, "constant/g"), G)
    write(os.path.join(case, "constant/thermophysicalProperties"), THERMO)
    write(os.path.join(case, "constant/turbulenceProperties"), TURB)
    write(os.path.join(case, "system/controlDict"),
          head("dictionary", "controlDict", "system")
          + CONTROL_BODY.format(end=args.iters, wi=args.iters))
    write(os.path.join(case, "system/fvSchemes"), FVSCHEMES)
    write(os.path.join(case, "system/fvSolution"),
          FVSOLUTION.replace("{relax}", str(args.relax)))
    write(os.path.join(case, "system/meshQualityDict"), MESHQUALITY)
    write(os.path.join(case, "system/decomposeParDict"),
          head("dictionary", "decomposeParDict", "system")
          + DECOMPOSE_BODY.format(n=args.procs))
    write(os.path.join(case, "system/fvOptions"), FVOPTIONS)

    lo = np.array([-0.2, -0.3, -0.2])
    hi = np.array([9.0, 6.3, 3.4])
    write(os.path.join(case, "system/blockMeshDict"),
          block_mesh_dict(lo, hi, args.cell))
    write(os.path.join(case, "system/snappyHexMeshDict"),
          snappy_dict(patches, args.inside, args.level, args.vent_level,
                      args.leak_level))

    # Stamp the layout into the case. solve.py reads vent centres from here,
    # so a case can never be solved against another layout's geometry -- which
    # is exactly what happened when solve.py's --geometry default (cfd/geometry)
    # was used for a case built from cfd/geometry_L01: every supply face landed
    # a metre from the assumed centre, fell outside the profile's radius table,
    # and np.interp clamped it to the tail value. The case ran happily at 55% of
    # the intended supply flow.
    os.makedirs(os.path.join(case, "constant"), exist_ok=True)
    shutil.copy(os.path.join(args.geometry, "patches.json"),
                os.path.join(case, "constant", "patches.json"))
    tri = os.path.join(case, "constant/triSurface")
    os.makedirs(tri, exist_ok=True)
    shutil.copy(os.path.join(args.geometry, "room.stl"),
                os.path.join(tri, "room.stl"))
    n = [int(round((hi[i] - lo[i]) / args.cell)) for i in range(3)]
    print(f"wrote mesh setup to {case}: background {n[0]}x{n[1]}x{n[2]} "
          f"= {np.prod(n):,} cells at {args.cell} m, surface level {args.level}, "
          f"vents {args.vent_level}")




def patch_face_centres(case):
    """Face centres per boundary patch, straight from constant/polyMesh.

    Needed because this OpenFOAM build cannot compile `codedFixedValue`: its
    wmake rules carry no compiler (`/bin/sh: 1: -m64: not found`), so runtime
    code generation fails. Evaluating the vent profile here and writing static
    `nonuniform` lists avoids compilation entirely -- and is arguably better,
    since `inletOutlet` then switches on the solved flux rather than on the
    sign the profile table happened to have.
    """
    mesh = os.path.join(case, "constant/polyMesh")
    txt = open(os.path.join(mesh, "points")).read()
    pts = np.array(re.findall(
        r"\(\s*(-?[\d.eE+-]+)\s+(-?[\d.eE+-]+)\s+(-?[\d.eE+-]+)\s*\)", txt),
        dtype=float)
    txt = open(os.path.join(mesh, "faces")).read()
    faces = [np.fromstring(v, sep=" ", dtype=int)
             for _, v in re.findall(r"(\d+)\(([\d\s]+)\)", txt)]
    txt = open(os.path.join(mesh, "boundary")).read()
    out = {}
    for name, body in re.findall(r"^    (\w+)\n    \{(.*?)\n    \}", txt,
                                 re.S | re.M):
        n = int(re.search(r"nFaces\s+(\d+);", body).group(1))
        s0 = int(re.search(r"startFace\s+(\d+);", body).group(1))
        out[name] = np.array([pts[faces[i]].mean(axis=0)
                              for i in range(s0, s0 + n)])
    return out


def nonuniform(values, kind):
    rows = ("\n".join(f"({v[0]:.6g} {v[1]:.6g} {v[2]:.6g})" for v in values)
            if kind == "vector" else
            "\n".join(f"{v:.6g}" for v in values))
    return f"nonuniform List<{kind}>\n{len(values)}\n(\n{rows}\n)"


# --- 0/ ----------------------------------------------------------------------
#
# Wall thermal BCs come from case_spec.json. Surfaces it does not list -- the
# switched-off AC unit and the door frame -- are left ADIABATIC (zeroGradient)
# rather than given an invented temperature: a made-up fixedValue is a heat
# source, while zeroGradient at least adds no energy the data does not support.

INLET_K = 0.03          # 5% turbulence on the ~3 m/s diffuser jet
INLET_EPS = 0.03        # Cmu^0.75 k^1.5 / (0.07 * vent width)


def field(cls, obj, dims, internal, entries):
    body = "\n".join(entries)
    return (head(cls, obj, "0") + f"""
dimensions      {dims};
internalField   uniform {internal};

boundaryField
{{
{body}
    background
    {{
        type            zeroGradient;
    }}
}}
""")


def simple(patch, spec):
    return "    " + patch + "\n    {\n" + "".join(
        f"        {k:<15} {v};\n" for k, v in spec.items()) + "    }\n"


def stage_fields(args, patches, spec):
    import vent_bc
    case = args.out
    names = re.findall(r"^    (\w+)$",
                       open(os.path.join(case, "constant/polyMesh/boundary")).read(),
                       re.M)
    prof = json.load(open(args.profile))
    tab = dict(n=len(prof["r"]), rtab=vent_bc.fmt(prof["r"]),
               uztab=vent_bc.fmt(prof["u_z"]), urtab=vent_bc.fmt(prof["u_r"]),
               dttab=vent_bc.fmt(prof["dT"]))
    wsup = abs(spec["supply"]["w_m_s"])
    tsup = spec["supply"]["T_K"]
    twall = {k[:-2]: v for k, v in spec["thermal_bcs_UNKNOWN"].items()
             if k.endswith("_K")}
    leak_q = spec["leak"].get("measured_flow_m3_s", 0.0)

    centres = patch_face_centres(case)
    U, T, prgh, P, K, E, NUT, ALPHAT = ([] for _ in range(8))
    for n in sorted(names):
        role = patches.get(n, {}).get("role", "wall")
        if role == "supply":
            lo, hi = patches[n]["bbox_m"]
            cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
            c = centres[n]
            dx, dy = c[:, 0] - cx, c[:, 1] - cy
            r = np.hypot(dx, dy)
            inv = np.where(r > 1e-9, 1.0 / np.maximum(r, 1e-30), 0.0)
            uz = np.interp(r, prof["r"], prof["u_z"]) * wsup
            ur = np.interp(r, prof["r"], prof["u_r"]) * wsup
            vecs = np.c_[ur * dx * inv, ur * dy * inv, uz]
            tin = tsup + np.interp(r, prof["r"], prof["dT"])
            U.append(simple(n, {"type": "fixedValue",
                                "value": nonuniform(vecs, "vector")}))
            T.append(simple(n, {"type": "inletOutlet",
                                "inletValue": nonuniform(tin, "scalar"),
                                "value": nonuniform(tin, "scalar")}))
            prgh.append(simple(n, {"type": "fixedFluxPressure",
                                   "value": "uniform 0"}))
            K.append(simple(n, {"type": "fixedValue",
                                "value": f"uniform {INLET_K}"}))
            E.append(simple(n, {"type": "fixedValue",
                                "value": f"uniform {INLET_EPS}"}))
        elif role == "return":
            U.append(simple(n, {"type": "pressureInletOutletVelocity",
                                "value": "uniform (0 0 0)"}))
            T.append(simple(n, {"type": "inletOutlet",
                                "inletValue": f"uniform {tsup}",
                                "value": f"uniform {tsup}"}))
            prgh.append(simple(n, {"type": "prghPressure", "p": "uniform 0",
                                   "value": "uniform 0"}))
            K.append(simple(n, {"type": "inletOutlet",
                                "inletValue": f"uniform {INLET_K}",
                                "value": f"uniform {INLET_K}"}))
            E.append(simple(n, {"type": "inletOutlet",
                                "inletValue": f"uniform {INLET_EPS}",
                                "value": f"uniform {INLET_EPS}"}))
        elif role == "leak":
            # The MEASURED flux, and nothing else. This BC has now been wrong in
            # both directions, so the reasoning is worth keeping:
            #
            #   1. Prescribing 0.232 m3/s, from case_spec's supply-minus-return
            #      imbalance, was 258x too much: it forced 15.5 m/s through a
            #      10 mm slot and dragged the room toward the door (mean speed
            #      1.20 m/s against a reference 0.197).
            #   2. Making it a pressure outlet was still ~34x too much: 2.3 m/s
            #      through the slot, and every cell in the room faster than
            #      1 m/s was inside the undercut. That approach assumes the leak
            #      and the returns sit at the same pressure, which is false --
            #      the returns are fan-driven, the undercut opens onto a
            #      corridor at essentially room pressure -- so a shared p_rgh
            #      hands the leak a share set by conductance alone.
            #
            # Leak.csv measures it directly: 0.0678 m/s over 0.01331 m2 =
            # 0.0009 m3/s, 0.15% of supply. That is a measurement, not the
            # unreliable vent imbalance that caused (1).
            U.append(simple(n, {"type": "flowRateOutletVelocity",
                                "volumetricFlowRate": f"{leak_q}",
                                "value": "uniform (0 0 0)"}))
            T.append(simple(n, {"type": "zeroGradient"}))
            prgh.append(simple(n, {"type": "fixedFluxPressure",
                                   "value": "uniform 0"}))
            K.append(simple(n, {"type": "inletOutlet",
                                "inletValue": f"uniform {INLET_K}",
                                "value": f"uniform {INLET_K}"}))
            E.append(simple(n, {"type": "inletOutlet",
                                "inletValue": f"uniform {INLET_EPS}",
                                "value": f"uniform {INLET_EPS}"}))
        else:
            U.append(simple(n, {"type": "noSlip"}))
            if n in twall:
                T.append(simple(n, {"type": "fixedValue",
                                    "value": f"uniform {twall[n]}"}))
            else:
                T.append(simple(n, {"type": "zeroGradient"}))
            prgh.append(simple(n, {"type": "fixedFluxPressure",
                                   "value": "uniform 0"}))
            K.append(simple(n, {"type": "kqRWallFunction",
                                "value": f"uniform {INLET_K}"}))
            E.append(simple(n, {"type": "epsilonWallFunction",
                                "value": f"uniform {INLET_EPS}"}))
        P.append(simple(n, {"type": "calculated", "value": "uniform 101325"}))
        NUT.append(simple(n, {"type": "nutkWallFunction", "value": "uniform 0"}
                          if role == "wall" else
                          {"type": "calculated", "value": "uniform 0"}))
        ALPHAT.append(simple(n, {"type": "compressible::alphatWallFunction",
                                 "Prt": "0.85", "value": "uniform 0"}
                             if role == "wall" else
                             {"type": "calculated", "value": "uniform 0"}))

    z = os.path.join(case, "0")
    write(os.path.join(z, "U"), field("volVectorField", "U",
          "[0 1 -1 0 0 0 0]", "(0 0 0)", U))
    write(os.path.join(z, "T"), field("volScalarField", "T",
          "[0 0 0 1 0 0 0]", f"{spec['return']['T_K_observed']}", T))
    write(os.path.join(z, "p_rgh"), field("volScalarField", "p_rgh",
          "[1 -1 -2 0 0 0 0]", "0", prgh))
    write(os.path.join(z, "p"), field("volScalarField", "p",
          "[1 -1 -2 0 0 0 0]", "101325", P))
    write(os.path.join(z, "k"), field("volScalarField", "k",
          "[0 2 -2 0 0 0 0]", f"{INLET_K}", K))
    write(os.path.join(z, "epsilon"), field("volScalarField", "epsilon",
          "[0 2 -3 0 0 0 0]", f"{INLET_EPS}", E))
    write(os.path.join(z, "nut"), field("volScalarField", "nut",
          "[0 2 -1 0 0 0 0]", "0", NUT))
    write(os.path.join(z, "alphat"), field("volScalarField", "alphat",
          "[1 -1 -1 0 0 0 0]", "0", ALPHAT))
    sup = [n for n in names if patches.get(n, {}).get("role") == "supply"]
    adiabatic = [n for n in names if patches.get(n, {}).get("role", "wall") == "wall"
                 and n not in twall]
    print(f"wrote 0/ for {len(names)} patches: {len(sup)} supply (coded profile), "
          f"leak prescribed at the measured {leak_q} m3/s "
          f"({leak_q/0.01331:.3f} m/s through the slot)")
    print(f"  adiabatic (no measured temperature): {', '.join(adiabatic)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cfd/run/base")
    ap.add_argument("--geometry", default="cfd/geometry")
    ap.add_argument("--stage", choices=("mesh", "fields", "all"), default="mesh")
    ap.add_argument("--cell", type=float, default=0.10, help="background cell [m]")
    ap.add_argument("--level", type=int, default=1, help="surface refinement")
    ap.add_argument("--vent-level", type=int, default=3, help="vent refinement")
    ap.add_argument("--leak-level", type=int, default=5,
                    help="leak refinement; the slot is 10 mm so it needs ~3 mm "
                         "cells or the patch is silently lost")
    ap.add_argument("--inside", type=float, nargs=3, default=[4.4, 3.0, 1.5])
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--relax", type=float, default=0.10,
                    help="relaxation for p_rgh, U, h and k/epsilon alike")
    ap.add_argument("--procs", type=int, default=8,
                    help="subdomains for decomposeParDict")
    ap.add_argument("--profile", default="cfd/vent_profile.json")
    ap.add_argument("--spec", default="cfd/case_spec.json")
    args = ap.parse_args()

    patches = json.load(open(os.path.join(args.geometry, "patches.json")))["patches"]
    if args.stage in ("mesh", "all"):
        stage_mesh(args, patches)
    if args.stage in ("fields", "all"):
        stage_fields(args, patches, json.load(open(args.spec)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
