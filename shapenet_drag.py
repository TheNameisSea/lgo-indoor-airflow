#!/usr/bin/env python
"""Drag coefficient on ShapeNet-Car, transcribed from the published implementation.

⚠ RUN THIS WITH cfdEnv, NOT ginotEnv:
    /path/to/cfd-env/bin/python shapenet_drag.py --validate

`vtk` is installed in cfdEnv (9.6.1) and absent from ginotEnv. That is a nuisance
but it is the RIGHT nuisance: the alternative is re-deriving `vtkPolyDataNormals`
with `AutoOrientNormals` by hand, and a sign error in the cell normals flips the
drag contribution of every face it touches while still producing a plausible
number. Use their filter, not our arithmetic.

WHY A TRANSCRIPTION AND NOT AN IMPLEMENTATION. The point of this file is to
reproduce `cd.txt`, the ground truth shipped with the raw release. Every
deviation is a candidate explanation for a mismatch, so there are none: the
formulas below are copied from `Car-Design-ShapeNetCar/utils/drag_coefficient.py`
line for line, including the parts that look odd (see the quirks section). Only
the file paths are ours.

THE GATE THIS EXISTS FOR. Before any model-predicted drag can mean anything, the
pipeline must reproduce the published C_D from GROUND-TRUTH fields. If it cannot,
nothing downstream is interpretable. That is `--validate`, and it costs no GPU.

⚠ QUIRKS OF THE PUBLISHED CODE, PRESERVED DELIBERATELY. None of these are bugs we
are fixing; changing any of them would mean we are no longer computing their
number:

  1. `velo_surf` is looked up from the VOLUME mesh by EXACT coordinate match
     (`{tuple(p): velo[i]}`), and any surface point absent from the volume mesh
     gets **zeros**. Only **994** of the
     3,682 surface points coincide with volume points. So ~73% of surface cells
     contribute a zero velocity gradient to the shear term. That is their
     behaviour, and it is preserved.
  2. `grad_u` per cell is `du_dx + du_dy + du_dz`, each a 3-vector formed by
     differencing the quad's four corner velocities over a norm of summed edge
     vectors. It is not a Jacobian and the sum of three directional derivatives
     is not a standard quantity; only its z-component is used.
  3. `drag_force = np.sum(pressure_force_component + wall_shear_stress_component)`
     where both terms are already scalars from `np.dot`, so the `np.sum` is a
     no-op.
  4. `A` is `ConvexHull(points[:, :2]).volume`. In 2-D scipy's `.volume` IS the
     enclosed area (`.area` would be the perimeter), and x-y is the plane normal
     to the flow, which we confirmed independently: mean volume velocity is
     [0.004, -0.043, 17.04], i.e. flow along +z, which is also why their code
     indexes `[:, -1]` for both normals and gradients.

CONSTANTS, from their `cal_coefficient`: inlet speed nu = 72/3.6 = 20 m/s,
air_density = 0.3, dynamic viscosity = 1.8e-5.
"""

import argparse
import os
import sys

import numpy as np
from scipy.spatial import ConvexHull

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

NU = 72 / 3.6
AIR_DENSITY = 0.3
DYNAMIC_VISCOSITY = 1.8e-5


def _vtk():
    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
    except ImportError:
        raise SystemExit(
            "[drag] vtk is not importable. Run this with cfdEnv:\n"
            "  /path/to/cfd-env/bin/python shapenet_drag.py --validate")
    return vtk, vtk_to_numpy


# --------------------------------------------------------------------------- #
#  Transcribed verbatim from utils/drag_coefficient.py
# --------------------------------------------------------------------------- #
def unstructured_grid_data_to_poly_data(ug):
    vtk, _ = _vtk()
    f = vtk.vtkDataSetSurfaceFilter()
    f.SetInputData(ug)
    f.Update()
    return f.GetOutput(), f


def load_unstructured_grid_data(file_name):
    vtk, _ = _vtk()
    r = vtk.vtkUnstructuredGridReader()
    r.SetFileName(file_name)
    r.Update()
    return r.GetOutput()


def calculate_pos(pos):
    """Frontal area: convex hull of the x-y projection (the plane normal to +z)."""
    return ConvexHull(pos[:, :2]).volume


def calculate_mesh_cell_area(ug):
    vtk, _ = _vtk()
    poly_data, _f = unstructured_grid_data_to_poly_data(ug)
    points = poly_data.GetPoints()
    cells = poly_data.GetPolys()
    cell_areas = np.zeros(cells.GetNumberOfCells())
    cells.InitTraversal()
    cell = vtk.vtkIdList()
    idx = 0
    while cells.GetNextCell(cell):
        if cell.GetNumberOfIds() == 4:
            p1 = np.array(points.GetPoint(cell.GetId(0)))
            p2 = np.array(points.GetPoint(cell.GetId(1)))
            p3 = np.array(points.GetPoint(cell.GetId(2)))
            p4 = np.array(points.GetPoint(cell.GetId(3)))
            area = 0.5 * (np.linalg.norm(np.cross(p2 - p1, p3 - p1)) +
                          np.linalg.norm(np.cross(p3 - p1, p4 - p1)))
            cell_areas[idx] += area
            idx += 1
    return cell_areas


def calculate_cell_velocity_gradient(ug, velocity):
    vtk, _ = _vtk()
    velocity_data = vtk.vtkDoubleArray()
    velocity_data.SetNumberOfComponents(3)
    velocity_data.SetNumberOfTuples(ug.GetNumberOfPoints())
    velocity_data.SetName("Velocity")
    for i in range(ug.GetNumberOfPoints()):
        velocity_data.SetTuple(i, velocity[i])
    ug.GetPointData().AddArray(velocity_data)

    poly_data, _f = unstructured_grid_data_to_poly_data(ug)
    points = poly_data.GetPoints()
    grad_u = np.zeros((poly_data.GetNumberOfCells(), 3))
    cells = poly_data.GetPolys()
    cells.InitTraversal()
    cell = vtk.vtkIdList()
    idx = 0
    arr = poly_data.GetPointData().GetArray("Velocity")
    while cells.GetNextCell(cell):
        if cell.GetNumberOfIds() == 4:
            p1 = np.array(points.GetPoint(cell.GetId(0)))
            p2 = np.array(points.GetPoint(cell.GetId(1)))
            p3 = np.array(points.GetPoint(cell.GetId(2)))
            p4 = np.array(points.GetPoint(cell.GetId(3)))
            u1 = np.array(arr.GetTuple(cell.GetId(0)))
            u2 = np.array(arr.GetTuple(cell.GetId(1)))
            u3 = np.array(arr.GetTuple(cell.GetId(2)))
            u4 = np.array(arr.GetTuple(cell.GetId(3)))
            du_dx = (u2 - u1 + u3 - u4) / (np.linalg.norm(p2 - p1 + p3 - p4) + 1e-8)
            du_dy = (u3 - u1 + u4 - u2) / (np.linalg.norm(p3 - p1 + p4 - p2) + 1e-8)
            du_dz = (u4 - u1 + u2 - u3) / (np.linalg.norm(p4 - p1 + p2 - p3) + 1e-8)
            grad_u[idx] += (du_dx + du_dy + du_dz)
            idx += 1
    return grad_u


def calculate_drag_force(cell_areas, surface_normals, pressure_array,
                         velocity_gradients, dynamic_viscosity):
    pressure_force_component = -np.dot(
        pressure_array.flatten() * cell_areas.flatten(), surface_normals.flatten())
    wall_shear_stress_component = -np.dot(
        velocity_gradients.flatten() * cell_areas.flatten(),
        surface_normals.flatten()) * dynamic_viscosity
    return np.sum(pressure_force_component + wall_shear_stress_component)


def get_normal(ug):
    vtk, vtk_to_numpy = _vtk()
    poly_data, _f = unstructured_grid_data_to_poly_data(ug)
    nf = vtk.vtkPolyDataNormals()
    nf.SetInputData(poly_data)
    nf.SetAutoOrientNormals(1)
    nf.SetConsistency(1)
    nf.SetComputeCellNormals(1)
    nf.SetComputePointNormals(0)
    nf.Update()
    return vtk_to_numpy(nf.GetOutput().GetCellData().GetNormals())


# --------------------------------------------------------------------------- #
#  Our entry point: same computation, our paths
# --------------------------------------------------------------------------- #
def cd_for_case(case_dir_path, press_surf=None, velo_surf=None):
    """C_D for one raw case directory.

    `press_surf` / `velo_surf` are POINT arrays on the 3,682 surface points.
    Leave both None for the ground-truth computation (the `--validate` path);
    pass predictions to score a model.
    """
    vtk, vtk_to_numpy = _vtk()
    f_press = os.path.join(case_dir_path, "quadpress_smpl.vtk")
    f_velo = os.path.join(case_dir_path, "hexvelo_smpl.vtk")
    ug_press = load_unstructured_grid_data(f_press)
    ug_velo = load_unstructured_grid_data(f_velo)

    normal_surf = get_normal(ug_press)
    points_surf = vtk_to_numpy(ug_press.GetPoints().GetData())
    A = calculate_pos(points_surf)
    cell_areas = calculate_mesh_cell_area(ug_press)

    if velo_surf is None:
        # QUIRK 1: exact-coordinate lookup into the volume mesh; misses get zeros.
        velo = vtk_to_numpy(ug_velo.GetPointData().GetVectors())
        points_velo = vtk_to_numpy(ug_velo.GetPoints().GetData())
        velo_dict = {tuple(p): velo[i] for i, p in enumerate(points_velo)}
        velo_surf = np.array([velo_dict[tuple(p)] if tuple(p) in velo_dict
                              else np.zeros(3) for p in points_surf])

    grad_u = calculate_cell_velocity_gradient(ug_press, velo_surf)

    if press_surf is None:
        c2p = vtk.vtkPointDataToCellData()
        c2p.SetInputData(ug_press)
        c2p.Update()
        press_cell = vtk_to_numpy(c2p.GetOutput().GetCellData().GetScalars())
    else:
        press_data = vtk.vtkDoubleArray()
        press_data.SetNumberOfComponents(1)
        press_data.SetNumberOfTuples(ug_press.GetNumberOfPoints())
        press_data.SetName("my_press")
        for i in range(ug_press.GetNumberOfPoints()):
            press_data.SetTuple(i, [float(press_surf[i])])
        ug_press.GetPointData().AddArray(press_data)
        c2p = vtk.vtkPointDataToCellData()
        c2p.SetInputData(ug_press)
        c2p.Update()
        press_cell = vtk_to_numpy(c2p.GetOutput().GetCellData().GetArray("my_press"))

    drag_force = calculate_drag_force(cell_areas, normal_surf[:, -1], press_cell,
                                      grad_u[:, -1], np.array(DYNAMIC_VISCOSITY))
    return (2 / ((NU ** 2) * A * AIR_DENSITY)) * drag_force


def validate(n_cases=8):
    """THE GATE: ground-truth fields must reproduce the shipped `cd.txt`."""
    from data.shapenet_raw import case_dir, drag, mapping_all
    m = mapping_all()
    ids = sorted(m)[:n_cases]
    rows = []
    for nid in ids:
        d = case_dir(nid, m)
        ours = cd_for_case(d)
        theirs = drag(nid, m)
        rows.append((nid, ours, theirs))
        print(f"  {nid}  ours {ours:+.6f}   cd.txt {theirs:+.6f}   "
              f"diff {ours - theirs:+.2e}  rel {abs(ours-theirs)/abs(theirs):.3%}")
    o = np.array([r[1] for r in rows])
    t = np.array([r[2] for r in rows])
    rel = np.abs(o - t) / np.abs(t)
    print(f"\n[drag] n={len(rows)}  max rel err {rel.max():.3%}  "
          f"mean {rel.mean():.3%}  corr {np.corrcoef(o, t)[0,1]:+.6f}")
    if rel.max() < 0.01:
        print("[drag] GATE PASS — the pipeline reproduces cd.txt.")
    else:
        print("[drag] GATE FAIL — do NOT compute model drag until this is resolved.")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    if a.validate:
        validate(a.n)
    else:
        ap.print_help()
