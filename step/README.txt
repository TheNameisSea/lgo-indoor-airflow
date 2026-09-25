step/H105.step — the CAD source for the room geometry.

The per-case STL patches are NOT shipped (~120 MB, regenerable). Rebuild them:

    python cfd/step_to_stl.py

which writes cfd/geometry/*.stl. cfd/make_layout.py then places the six vent
panels for a given layout, and cfd/make_case.py assembles the OpenFOAM case.
