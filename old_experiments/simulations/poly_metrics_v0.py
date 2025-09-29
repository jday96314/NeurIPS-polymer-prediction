# Auto-generated from feature_exploration.ipynb (ported to a clean module)
# Ubuntu + Python >= 3.11. Verbose variable names. Single spaces around equals.
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional, Tuple, Dict

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import msd, polymer
from MDAnalysis.lib import mdamath
import freud
import pickle
import joblib
from rdkit.Chem import Descriptors3D
from tqdm import tqdm
import math
from MDAnalysis.exceptions import NoDataError

# -----------------------------
# LAMMPS I/O + topology helpers
# -----------------------------

def parse_type_to_mass_map_from_lammps_data(data_file_path: str) -> dict[int, float]:
    """
    Parse the 'Masses' section of a LAMMPS DATA file and return {atom_type: mass_amu}.
    """
    with open(data_file_path, "r") as file_handle:
        all_lines = file_handle.read().splitlines()

    start_index = None
    for index, line_text in enumerate(all_lines):
        if line_text.strip().lower().startswith("masses"):
            start_index = index + 1
            break
    if start_index is None:
        raise ValueError("No 'Masses' section found in DATA file.")

    type_to_mass_map: dict[int, float] = {}
    numeric_line_pattern = re.compile(
        r"^\s*(\d+)\s+([+\-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+\-]?\d+)?)"
    )

    i = start_index
    while i < len(all_lines):
        text = all_lines[i].strip()
        i += 1
        if not text:
            continue
        if text[0].isalpha():  # next section header like "Atoms", "Bonds", etc.
            break
        m = numeric_line_pattern.match(text)
        if m:
            atom_type = int(m.group(1))
            mass_amu = float(m.group(2))
            type_to_mass_map[atom_type] = mass_amu

    if not type_to_mass_map:
        raise ValueError("Failed to parse any masses from the 'Masses' section.")
    return type_to_mass_map


def initialize_universe_with_dump_and_masses(
    lammps_dump_path: str,
    lammps_data_path_for_masses: str,
) -> mda.Universe:
    """
    Load coordinates/box/types from a LAMMPS dump, attach masses from a DATA file's 'Masses' section,
    and return a ready-to-use MDAnalysis Universe.
    """
    universe = mda.Universe(lammps_dump_path, format="LAMMPSDUMP")

    type_to_mass_amu = parse_type_to_mass_map_from_lammps_data(lammps_data_path_for_masses)

    atom_types_raw = np.asarray(universe.atoms.types)
    try:
        atom_types_int = atom_types_raw.astype(int)
    except Exception:
        atom_types_int = np.array([int(str(t)) for t in atom_types_raw], dtype=int)

    missing_types = sorted(set(atom_types_int) - set(type_to_mass_amu.keys()))
    if missing_types:
        raise KeyError(f"Missing masses for atom types: {missing_types}")

    per_atom_masses_amu = np.array([type_to_mass_amu[t] for t in atom_types_int], dtype=float)
    universe.add_TopologyAttr("masses", per_atom_masses_amu)

    average_atomic_mass_amu = float(per_atom_masses_amu.mean())
    if not (3.0 <= average_atomic_mass_amu <= 40.0):
        raise RuntimeError(f"Average atomic mass looks off: {average_atomic_mass_amu:.2f} amu")

    return universe


def parse_bond_atom_ids_from_lammps_data(data_file_path: str) -> np.ndarray:
    """
    Returns an (N, 2) array of LAMMPS atom IDs (1-based) from the DATA file's 'Bonds' section.
    Assumes canonical 'Bonds' lines:  id  type  atom_i  atom_j  [# comment]
    """
    with open(data_file_path, "r") as file_handle:
        lines = file_handle.read().splitlines()

    start_index = None
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("bonds"):
            start_index = i + 1
            break
    if start_index is None:
        raise ValueError("No 'Bonds' section found in DATA file.")

    bonds: list[tuple[int, int]] = []
    line_pattern = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)")
    i = start_index
    while i < len(lines):
        text = lines[i].strip()
        i += 1
        if not text:
            continue
        if text[0].isalpha():  # next section header
            break
        m = line_pattern.match(text)
        if m:
            ai_id = int(m.group(3))  # LAMMPS atom IDs (1-based)
            aj_id = int(m.group(4))
            if ai_id != aj_id:
                bonds.append((ai_id, aj_id))
    if not bonds:
        raise ValueError("Parsed zero bonds from 'Bonds' section.")
    return np.asarray(bonds, dtype=int)


def attach_bonds_by_atom_ids(universe: mda.Universe, bonds_atom_ids_1based: np.ndarray) -> None:
    """
    Map (atom_id_i, atom_id_j) to 0-based indices in the current Universe and attach as 'bonds'.
    """
    id_to_index = {int(atom_id): idx for idx, atom_id in enumerate(universe.atoms.ids)}
    pairs = []
    for ai_id, aj_id in bonds_atom_ids_1based:
        if ai_id in id_to_index and aj_id in id_to_index:
            ai = id_to_index[ai_id]
            aj = id_to_index[aj_id]
            if ai != aj:
                pairs.append((ai, aj))
    if not pairs:
        raise RuntimeError("No bonds matched current atom IDs; check that DATA and dump refer to the same system.")
    bonds_indices_0based = np.asarray(pairs, dtype=int)
    if getattr(universe, "bonds", None) is None or len(universe.bonds) == 0:
        universe.add_TopologyAttr("bonds", bonds_indices_0based)


# -----------------------------
# FFV (Monte Carlo, freud)
# -----------------------------

'''
_BONDI_VDW_RADIUS_BY_ELEMENT_ANGSTROM: dict[str, float] = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47,
    "P": 1.80, "S": 1.80, "Cl": 1.75, "Br": 1.85, "Si": 2.10,
}


def build_per_atom_vdw_radii_angstrom(universe: mda.Universe) -> np.ndarray:
    """
    Derive reasonable vdW radii per atom from element symbols or atom names.
    Falls back to a generic heavy-atom radius if needed.
    """
    try:
        element_symbols = [el if el is not None else "" for el in getattr(universe.atoms, "elements", [None]*len(universe.atoms))]
    except Exception:
        element_symbols = ["" for _ in range(len(universe.atoms))]

    per_atom_radii: list[float] = []
    for atom, element_symbol in zip(universe.atoms, element_symbols):
        key = element_symbol if element_symbol in _BONDI_VDW_RADIUS_BY_ELEMENT_ANGSTROM else str(getattr(atom, "name", "")).rstrip("0123456789")
        if key in _BONDI_VDW_RADIUS_BY_ELEMENT_ANGSTROM:
            radius = _BONDI_VDW_RADIUS_BY_ELEMENT_ANGSTROM[key]
        elif key and key[0] in _BONDI_VDW_RADIUS_BY_ELEMENT_ANGSTROM:
            radius = _BONDI_VDW_RADIUS_BY_ELEMENT_ANGSTROM[key[0]]
        else:
            radius = 1.70  # generic heavy atom
        per_atom_radii.append(radius)
    return np.asarray(per_atom_radii, dtype=float)


def estimate_ffv_monte_carlo(
    universe: mda.Universe,
    n_samples: int = 200_000,
    probe_radius_angstrom: float = 1.2,
    seed: int = 0,
) -> tuple[float, float]:
    """
    Estimate probe-accessible fractional free volume (FFV) using random sampling in the unit cell.

    Returns (mean_ffv, std_estimate) where std_estimate is the binomial standard error.
    """
    rng = np.random.default_rng(seed)

    positions_angstrom = universe.atoms.positions.astype(np.float64)
    box_matrix_angstrom = mdamath.triclinic_vectors(universe.dimensions)  # (3,3)
    freud_box = freud.Box.from_matrix(box_matrix_angstrom)

    per_atom_vdw_radii_angstrom = build_per_atom_vdw_radii_angstrom(universe)

    # Sample fractional coords in [0,1)^3, map to Cartesian
    sample_fracs = rng.random((int(n_samples), 3), dtype=float)
    sample_points_angstrom = sample_fracs @ box_matrix_angstrom.T

    # Neighbor search with conservative cutoff
    max_effective_radius = float(np.max(per_atom_vdw_radii_angstrom) + probe_radius_angstrom)
    neighbor_query = freud.locality.AABBQuery(freud_box, positions_angstrom)
    neighbor_list = neighbor_query.query(sample_points_angstrom, {"r_max": max_effective_radius}).toNeighborList()

    occupied_flags = np.zeros(sample_points_angstrom.shape[0], dtype=bool)
    query_indices = np.asarray(neighbor_list.query_point_indices, dtype=np.int64)
    point_indices = np.asarray(neighbor_list.point_indices, dtype=np.int64)
    occupied_flags[query_indices] = True

    # Optional refinement by per-atom radii if separations are available
    try:
        neighbor_vectors = np.asarray(neighbor_list.separations, dtype=float)
        neighbor_distances = np.linalg.norm(neighbor_vectors, axis=1)
        effective_radii_per_pair = per_atom_vdw_radii_angstrom[point_indices] + probe_radius_angstrom
        within_true_radius = neighbor_distances <= (effective_radii_per_pair + 1e-9)
        occupied_flags[:] = False
        occupied_flags[query_indices[within_true_radius]] = True
    except Exception:
        pass

    ffv_fraction = float((~occupied_flags).mean())
    # Binomial standard error (quick uncertainty proxy)
    p = ffv_fraction
    std_err = float(np.sqrt(max(p * (1 - p), 0.0) / max(n_samples, 1)))
    return ffv_fraction, std_err
#'''

#'''
# Minimal Bondi vdW radii (Å). Add to taste if your system contains more elements.
BONDI_RADII_ANGSTROM: Dict[str, float] = {
    "H": 1.20, "He": 1.40,
    "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47,
    "P": 1.80, "S": 1.80, "Cl": 1.75,
    "Br": 1.85, "I": 1.98,
    "Si": 2.10,
    # Alkali/alkaline-earth (rare in organic polymers, included for completeness):
    "Na": 2.27, "K": 2.75, "Ca": 2.31,
}

APPROX_ELEMENT_MASSES = {
    "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998,
    "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.085, "P": 30.974,
    "S": 32.06, "Cl": 35.45, "K": 39.098, "Ca": 40.078, "Br": 79.904, "I": 126.904,
}

def safe_get(atom, attr: str):
    try:
        return getattr(atom, attr)
    except (AttributeError, NoDataError):
        return None

def infer_element_symbol_for_atom(atom) -> str:
    # 1) element
    el = safe_get(atom, "element")
    if el:
        return el.strip().capitalize()

    # 2) name
    nm = safe_get(atom, "name")
    if isinstance(nm, str) and nm:
        letters = "".join(c for c in nm if c.isalpha())
        if letters:
            return (letters[:2].capitalize()
                    if len(letters) > 1 and letters[1].islower()
                    else letters[:1].upper())

    # 3) type (LAMMPS often has numeric or short strings)
    tp = safe_get(atom, "type")
    if tp:
        if isinstance(tp, str):
            letters = "".join(c for c in tp if c.isalpha())
            if letters:
                return (letters[:2].capitalize()
                        if len(letters) > 1 and letters[1].islower()
                        else letters[:1].upper())
        # numeric types → just fall through to default after mass check

    # 4) mass-based nearest match (crude but better than nothing)
    mass = safe_get(atom, "mass")
    if mass:
        closest_el = min(APPROX_ELEMENT_MASSES, key=lambda k: abs(APPROX_ELEMENT_MASSES[k] - mass))
        return closest_el

    # 5) default
    print('Yikes 2!')
    return "C"

def compute_fractional_free_volume(
    universe: mda.Universe,
    selection: str = "all",
    grid_spacing_angstrom: float = 0.5,
    radii_angstrom_by_element: Optional[Dict[str, float]] = None,
    probe_radius_angstrom: Optional[float] = None,
    use_periodic_boundary_conditions: bool = True,
    return_occupancy_grid: bool = False,
) -> Tuple[dict, Optional[np.ndarray]]:
    """
    Estimate FFV for a selected set of atoms by voxelizing the periodic box and marking
    voxels whose centers lie within a vdW sphere of any atom (optionally inflated by a probe).

    Parameters
    ----------
    universe : mda.Universe
        Universe with box dimensions in Å. Assumes orthorhombic (alpha=beta=gamma≈90°).
    selection : str
        Atom selection (e.g., "all" or "not name H*").
    grid_spacing_angstrom : float
        Edge length of cubic voxels in Å. Smaller -> more accurate, slower. 0.5 Å is a good start.
    radii_angstrom_by_element : dict
        Mapping element symbol -> vdW radius in Å. Defaults to minimal Bondi set above.
    probe_radius_angstrom : float | None
        If provided, each atom's radius is inflated by this amount before painting occupancy.
        Set to 0.0 for strict vdW volume, or e.g. 1.2 Å for helium-probe-like accessible volume.
    use_periodic_boundary_conditions : bool
        If True, spheres wrap across boundaries. If False, atoms near edges will be truncated.
    return_occupancy_grid : bool
        If True, also return a boolean 3D array of occupied voxels (z, y, x).

    Returns
    -------
    results : dict
        {
          "ffv": float,
          "occupied_volume_A3": float,
          "box_volume_A3": float,
          "voxel_count_total": int,
          "voxel_count_occupied": int,
          "grid_shape_zyx": (nz, ny, nx),
          "grid_spacing_A": float,
          "selection": str,
          "notes": str,
        }
    occupancy_grid : np.ndarray[bool] | None
        Boolean 3D mask of occupied voxels (z, y, x) if return_occupancy_grid is True, else None.
    """
    radii_lookup = radii_angstrom_by_element or BONDI_RADII_ANGSTROM

    # Grab a single frame's dimensions (Å) and validate orthorhombic box.
    lx, ly, lz, alpha, beta, gamma = universe.trajectory.ts._unitcell  # MDAnalysis stores [lx, ly, lz, alpha, beta, gamma]
    def _is_close(a: float, b: float, tol: float = 1e-2) -> bool:
        return abs(a - b) < tol
    if not (_is_close(alpha, 90.0) and _is_close(beta, 90.0) and _is_close(gamma, 90.0)):
        raise ValueError("This FFV helper currently assumes an orthorhombic box (alpha=beta=gamma≈90°).")

    total_box_volume_A3 = float(lx * ly * lz)

    # Build grid (voxel centers). We ensure at least 1 voxel per axis.
    nx = max(1, int(math.floor(lx / grid_spacing_angstrom)))
    ny = max(1, int(math.floor(ly / grid_spacing_angstrom)))
    nz = max(1, int(math.floor(lz / grid_spacing_angstrom)))
    actual_dx = lx / nx
    actual_dy = ly / ny
    actual_dz = lz / nz

    # Occupancy grid, indexed as (z, y, x)
    occupancy_grid = np.zeros((nz, ny, nx), dtype=bool)

    selected_atoms = universe.select_atoms(selection)
    if selected_atoms.n_atoms == 0:
        raise ValueError(f"No atoms found for selection='{selection}'.")

    # Precompute voxel center coordinates along each axis (Å).
    # Centers are offset by half a voxel from box edges: [dx/2, 3dx/2, ...]
    x_centers = (np.arange(nx, dtype=float) + 0.5) * actual_dx
    y_centers = (np.arange(ny, dtype=float) + 0.5) * actual_dy
    z_centers = (np.arange(nz, dtype=float) + 0.5) * actual_dz

    # Helper to convert a coordinate (Å) to voxel index with optional periodic wrapping.
    def coord_to_index(coord: float, axis_length: float, ndiv: int) -> int:
        raw = int(math.floor(coord / (axis_length / ndiv)))
        if use_periodic_boundary_conditions:
            return raw % ndiv
        return min(max(raw, 0), ndiv - 1)

    # For faster distance checks, we will “paint” per atom by bounding-boxing voxels that could be inside the sphere,
    # then refine with a center-to-atom distance test.
    positions_A = selected_atoms.positions.copy()  # shape (N, 3) in Å
    # Wrap into the primary cell so indices work cleanly.
    if use_periodic_boundary_conditions:
        positions_A[:, 0] = np.mod(positions_A[:, 0], lx)
        positions_A[:, 1] = np.mod(positions_A[:, 1], ly)
        positions_A[:, 2] = np.mod(positions_A[:, 2], lz)

    # Acquire per-atom radii. Try atom.element when present; fall back to simple guesses from name/type.
    atom_radii = np.empty(selected_atoms.n_atoms, dtype=float)
    '''
    for i, atom in enumerate(selected_atoms.atoms):
        element_symbol = infer_element_symbol_for_atom(atom)

        # Normalize halogens like "CL" → "Cl"
        if len(element_symbol) >= 2:
            element_symbol = element_symbol[0].upper() + element_symbol[1].lower()
        else:
            element_symbol = element_symbol.upper()

        vdW_radius = radii_lookup.get(element_symbol, radii_lookup.get("C", 1.70))
        if probe_radius_angstrom is not None:
            vdW_radius += float(probe_radius_angstrom)
        atom_radii[i] = vdW_radius
    '''
    for i, atom in enumerate(selected_atoms.atoms):
        # Temporarily get the atom name for printing
        atom_name_for_debug = safe_get(atom, "name") or "N/A"

        element_symbol = infer_element_symbol_for_atom(atom)

        # Normalize halogens...
        if len(element_symbol) >= 2:
            element_symbol = element_symbol[0].upper() + element_symbol[1].lower()
        else:
            element_symbol = element_symbol.upper()

        vdW_radius = radii_lookup.get(element_symbol, radii_lookup.get("C", 1.70))

        if probe_radius_angstrom is not None:
            vdW_radius += float(probe_radius_angstrom)
        atom_radii[i] = vdW_radius

    # Precompute squared radii for distance tests.
    atom_radii_sq = atom_radii ** 2

    # Paint each atom's sphere onto the voxel grid
    for atom_index in range(selected_atoms.n_atoms):
        ax, ay, az = positions_A[atom_index]
        r = atom_radii[atom_index]

        # Determine voxel index bounds along each axis (inclusive) potentially covered by the sphere.
        min_x = ax - r
        max_x = ax + r
        min_y = ay - r
        max_y = ay + r
        min_z = az - r
        max_z = az + r

        ix_min = int(math.floor(min_x / actual_dx))
        ix_max = int(math.floor(max_x / actual_dx))
        iy_min = int(math.floor(min_y / actual_dy))
        iy_max = int(math.floor(max_y / actual_dy))
        iz_min = int(math.floor(min_z / actual_dz))
        iz_max = int(math.floor(max_z / actual_dz))

        # Build index ranges with optional periodic wrap
        def wrapped_range(i_min: int, i_max: int, ndiv: int) -> np.ndarray:
            if not use_periodic_boundary_conditions:
                i0 = max(i_min, 0)
                i1 = min(i_max, ndiv - 1)
                if i1 < i0:
                    return np.empty(0, dtype=int)
                return np.arange(i0, i1 + 1, dtype=int)
            # wrap: e.g., range(-2, 1) in ndiv=10 -> [8, 9, 0, 1]
            full = np.arange(i_min, i_max + 1, dtype=int)
            return np.mod(full, ndiv)

        x_indices = wrapped_range(ix_min, ix_max, nx)
        y_indices = wrapped_range(iy_min, iy_max, ny)
        z_indices = wrapped_range(iz_min, iz_max, nz)

        # Squared distance threshold
        r2 = atom_radii_sq[atom_index]

        # Distance test on voxel centers within the bounding box
        for iz in z_indices:
            zc = z_centers[iz]
            dz = zc - az
            if use_periodic_boundary_conditions:
                # Minimum-image along each axis (orthorhombic)
                if dz > 0.5 * lz:
                    dz -= lz
                elif dz < -0.5 * lz:
                    dz += lz
            dz2 = dz * dz
            # Quick prune if dz already exceeds r
            if dz2 > r2:
                continue

            for iy in y_indices:
                yc = y_centers[iy]
                dy = yc - ay
                if use_periodic_boundary_conditions:
                    if dy > 0.5 * ly:
                        dy -= ly
                    elif dy < -0.5 * ly:
                        dy += ly
                dy2 = dy * dy
                if dy2 + dz2 > r2:
                    continue

                # Compute remaining allowed dx span (sphere cross-section)
                remaining_r2 = r2 - (dy2 + dz2)

                # Instead of testing all x individually, we can still loop—vectorizing here would add complexity.
                for ix in x_indices:
                    xc = x_centers[ix]
                    dx = xc - ax
                    if use_periodic_boundary_conditions:
                        if dx > 0.5 * lx:
                            dx -= lx
                        elif dx < -0.5 * lx:
                            dx += lx
                    if dx * dx <= remaining_r2:
                        occupancy_grid[iz, iy, ix] = True

    voxel_volume_A3 = float(actual_dx * actual_dy * actual_dz)
    voxel_count_total = int(nx * ny * nz)
    voxel_count_occupied = int(np.count_nonzero(occupancy_grid))
    occupied_volume_A3 = float(voxel_count_occupied) * voxel_volume_A3

    ffv = max(0.0, min(1.0, 1.0 - (occupied_volume_A3 / total_box_volume_A3)))

    results = {
        "ffv": ffv,
        "occupied_volume_A3": occupied_volume_A3,
        "box_volume_A3": total_box_volume_A3,
        "voxel_count_total": voxel_count_total,
        "voxel_count_occupied": voxel_count_occupied,
        # "grid_shape_zyx": (nz, ny, nx),
        "grid_spacing_A": float(grid_spacing_angstrom),
        "selection": selection,
    }

    # return (results, occupancy_grid if return_occupancy_grid else None)
    return results
#'''
    
# -----------------------------
# Diffusivity + shape metrics
# -----------------------------

def compute_diffusivity(
    universe: mda.Universe,
    selection: str = "all",
    fit_points: int = 20,
    time_per_frame_ps: float | None = None
) -> tuple[float, float]:
    """
    Returns (D in Å^2/ps, D in cm^2/s) via MSD slope / 6.
    """
    msd_result = msd.EinsteinMSD(universe, select=selection).run()
    msd_values_A2 = np.asarray(msd_result.results.timeseries, dtype=float)
    n_frames = len(msd_values_A2)
    if n_frames < 2:
        raise RuntimeError("Not enough frames to fit MSD slope (need >= 2 frames).")

    time_values_ps = np.asarray(msd_result.times, dtype=float)
    if time_values_ps.size != n_frames or not np.all(np.isfinite(time_values_ps)):
        dt_ps = time_per_frame_ps if time_per_frame_ps is not None else getattr(universe.trajectory, "dt", None)
        if dt_ps is None:
            raise RuntimeError("No time axis; provide time_per_frame_ps (e.g., 0.2 for 2 fs step dumped every 100).")
        time_values_ps = np.arange(n_frames, dtype=float) * float(dt_ps)

    k = max(2, min(int(fit_points), n_frames))
    slope_A2_per_ps = float(np.polyfit(time_values_ps[:k], msd_values_A2[:k], 1)[0])
    diffusivity_A2_per_ps = slope_A2_per_ps / 6.0
    diffusivity_cm2_per_s = diffusivity_A2_per_ps * 1e-4
    return diffusivity_A2_per_ps, diffusivity_cm2_per_s


def mass_weighted_gyration_eigenvalues(atomgroup: mda.core.groups.AtomGroup) -> np.ndarray:
    """
    Return eigenvalues (λ1 >= λ2 >= λ3) of the mass-weighted gyration tensor (Å²).
    """
    positions_angstrom = atomgroup.positions.astype(float)
    masses_amu = atomgroup.masses.astype(float)
    if positions_angstrom.size == 0:
        raise ValueError("AtomGroup is empty.")
    if np.all(masses_amu == 0):
        raise ValueError("Masses are all zero; attach masses before computing shape metrics.")

    center_of_mass_angstrom = atomgroup.center_of_mass(wrap=True).astype(float)
    centered_positions = positions_angstrom - center_of_mass_angstrom

    total_mass = float(np.sum(masses_amu))
    weighted_positions = centered_positions * masses_amu[:, None]
    gyration_tensor_A2 = (weighted_positions.T @ centered_positions) / total_mass  # (3,3)

    eigenvalues = np.linalg.eigvalsh(gyration_tensor_A2)
    return eigenvalues[::-1].astype(float)


def shape_metrics_from_eigs(eigvals_desc: np.ndarray) -> dict[str, float]:
    lam1, lam2, lam3 = map(float, eigvals_desc)
    trace = lam1 + lam2 + lam3
    asphericity_A2 = lam1 - 0.5 * (lam2 + lam3)
    acylindricity_A2 = lam2 - lam3
    kappa2 = 1.0 - 3.0 * ((lam1*lam2 + lam2*lam3 + lam3*lam1) / (trace*trace + 1e-30))
    return {
        "asphericity_A2": asphericity_A2,
        "acylindricity_A2": acylindricity_A2,
        "kappa2": kappa2,
        "trace_A2": trace,
        "lambda1_A2": lam1,
        "lambda2_A2": lam2,
        "lambda3_A2": lam3,
    }


# -----------------------------
# Backbone graph + persistence
# -----------------------------

def _build_adjacency_from_bonds(n_atoms: int, bonds_0based: np.ndarray) -> list[list[int]]:
    adjacency = [[] for _ in range(n_atoms)]
    for ai, aj in bonds_0based:
        adjacency[ai].append(aj)
        adjacency[aj].append(ai)
    return adjacency


def _bfs_path(adjacency: list[list[int]], start: int, goal: int) -> list[int]:
    from collections import deque
    queue = deque([start]); parent = {start: None}
    while queue:
        v = queue.popleft()
        if v == goal:
            break
        for w in adjacency[v]:
            if w not in parent:
                parent[w] = v
                queue.append(w)
    if goal not in parent:
        return []
    path = []
    cur = goal
    while cur is not None:
        path.append(cur); cur = parent[cur]
    return path[::-1]


def _double_bfs_longest_path(adjacency: list[list[int]], nodes: list[int]) -> list[int]:
    """
    Approximate longest simple path in an (almost) chain-like component via two BFS sweeps.
    """
    from collections import deque
    node_set = set(nodes)

    def farthest(x: int) -> tuple[int, dict[int, int | None]]:
        q = deque([x]); parent = {x: None}; last = x
        while q:
            v = q.popleft(); last = v
            for w in adjacency[v]:
                if w not in parent and w in node_set:
                    parent[w] = v; q.append(w)
        return last, parent

    a = nodes[0]
    a, _ = farthest(a)
    b, parent = farthest(a)

    path = []
    cur = b
    while cur is not None:
        path.append(cur); cur = parent[cur]
    return path[::-1]


def build_backbone_paths_heavy_atoms(universe: mda.Universe, min_atoms_per_chain: int = 12) -> list[mda.core.groups.AtomGroup]:
    assert getattr(universe, "bonds", None) is not None and len(universe.bonds) > 0, "Universe has no bonds; attach them first."
    heavy_mask = (universe.atoms.masses > 1.2)
    heavy_indices = set(np.nonzero(heavy_mask)[0].tolist())
    adjacency = _build_adjacency_from_bonds(len(universe.atoms), universe.bonds.to_indices())

    # connected components within heavy subgraph
    from collections import deque
    unvisited = set(heavy_indices)
    chains: list[mda.core.groups.AtomGroup] = []
    while unvisited:
        seed = next(iter(unvisited))
        component: list[int] = []
        q = deque([seed]); unvisited.remove(seed)
        while q:
            v = q.popleft(); component.append(v)
            for w in adjacency[v]:
                if w in unvisited:
                    unvisited.remove(w); q.append(w)

        degree = {v: sum((nbr in component) for nbr in adjacency[v]) for v in component}
        endpoints = [v for v in component if degree[v] == 1]
        if len(endpoints) >= 2:
            best: list[int] = []
            endpoint_set = set(endpoints)
            for i in range(len(endpoints)):
                for j in range(i + 1, len(endpoints)):
                    path = _bfs_path(adjacency, endpoints[i], endpoints[j])
                    path = [p for p in path if p in component]
                    if len(path) > len(best):
                        best = path
            path_indices = best
        else:
            path_indices = _double_bfs_longest_path(adjacency, component)

        if len(path_indices) >= min_atoms_per_chain:
            chains.append(universe.atoms[path_indices])
    return chains


def compute_persistence_length_stats(universe: mda.Universe, backbone_chains: list[mda.core.groups.AtomGroup]) -> dict[str, float]:
    per_chain_lp: list[float] = []
    for chain_ag in backbone_chains:
        try:
            pl_calc = polymer.PersistenceLength([chain_ag]).run()
            lp_value = float(np.nanmean(pl_calc.results.lp))
            if np.isfinite(lp_value):
                per_chain_lp.append(lp_value)
        except Exception:
            continue
    if not per_chain_lp:
        raise RuntimeError("Persistence length failed for all chains.")
    arr = np.array(per_chain_lp, dtype=float)
    return {
        "median_persistence_length": float(np.median(arr)),
        "mean_persistence_length": float(np.mean(arr)),
        "std_persistence_length": float(np.std(arr)),
        "p10_persistence_length": float(np.percentile(arr, 10)),
        "p90_persistence_length": float(np.percentile(arr, 90)),
    }


# -----------------------------
# High-level API
# -----------------------------

def compute_polymer_metrics(
    results_directory_path: str,
    # time_per_frame_ps: float | None = None,
    rng_seed: int = 0,
) -> dict[str, Any]:
    '''
    # Universe + masses
    universe = initialize_universe_with_dump_and_masses(
        lammps_dump_path=chosen_dump,
        lammps_data_path_for_masses=chosen_data,
    )

    # Attach bonds (best-effort)
    try:
        bonds_atom_ids = parse_bond_atom_ids_from_lammps_data(chosen_data)
        attach_bonds_by_atom_ids(universe, bonds_atom_ids)
    except Exception:
        bonds_atom_ids = None  # continue without bond-dependent metrics
    '''

    universe = mda.Universe(
        f'{results_directory_path}/eq1_final.data',
        f'{results_directory_path}/eq1_short.xtc',
        # format='LAMMPS'
    )

    last_frame = universe.trajectory[-2]

    # Density (g/cm^3)
    amu_to_g = 1.66053906660e-24
    volume_angstrom3 = mdamath.box_volume(universe.dimensions)
    rho_g_cm3 = (universe.atoms.masses.sum() * amu_to_g) / (volume_angstrom3 * 1e-24)

    # FFV (Monte Carlo, probe-based)
    # ffv_fraction, ffv_std = estimate_ffv_monte_carlo(
    ffv_stats = compute_fractional_free_volume(
        universe=universe,
        # n_samples=10_000,
        # probe_radius_angstrom=1.2,
        # seed=rng_seed,
    )

    # Radius of gyration & shape metrics (global)
    gyr_eigs = mass_weighted_gyration_eigenvalues(universe.atoms)
    shape = shape_metrics_from_eigs(gyr_eigs)
    rg_global_angstrom = float(np.sqrt(np.sum(gyr_eigs)))

    # # Diffusivity proxy via MSD slope (Å^2/ps) -> cm^2/s
    # try:
    #     D_A2ps, D_cm2s = compute_diffusivity(universe, selection="all", time_per_frame_ps=time_per_frame_ps)
    # except Exception:
    #     D_A2ps, D_cm2s = float("nan"), float("nan")

    # Backbone & persistence length (requires bonds)
    persistence_stats = {}
    number_of_backbone_chains = 0
    try:
        backbone_chains = build_backbone_paths_heavy_atoms(universe, min_atoms_per_chain=12)
        number_of_backbone_chains = len(backbone_chains)
        if number_of_backbone_chains > 0:
            persistence_stats = compute_persistence_length_stats(universe, backbone_chains)
    except Exception:
        persistence_stats = {}

    # Alternative Rg estimation approach.
    mol = joblib.load(f'{results_directory_path}/mol.pkl')
    rg_mol = Descriptors3D.RadiusOfGyration(mol)

    homopoly = joblib.load(f'{results_directory_path}/homopoly.pkl')
    rg_homopoly = Descriptors3D.RadiusOfGyration(homopoly)

    return {
        "density_g_cm3": float(rho_g_cm3),
        **ffv_stats,
        "rg_angstrom": float(rg_global_angstrom),
        "rg_mol": rg_mol,
        "rg_homopoly": rg_homopoly,
        **shape,
        **persistence_stats,
        # "diffusivity_A2_per_ps": float(D_A2ps),
        # "diffusivity_cm2_per_s": float(D_cm2s),
        # "n_backbone_chains": int(number_of_backbone_chains),
        # "paths": {"dump": chosen_dump, "data": chosen_data},
    }


if __name__ == "__main__":
    import glob
    import polars as pl

    all_smiles = pl.read_csv('data/from_host/train.csv')['SMILES'].to_list()

    DATASET_NAME = 'host_hybrid'
    final_dump_filepaths = glob.glob(f'simulations/work_dirs/{DATASET_NAME}/*/eq1_final.dump')
    data_directory_paths = [path.replace('/eq1_final.dump', '') for path in final_dump_filepaths]

    records = []
    for data_directory_path in tqdm(data_directory_paths):
        metrics = compute_polymer_metrics(data_directory_path)
        smiles = all_smiles[int(data_directory_path.split('/')[-1])]
        metrics['SMILES'] = smiles
        records.append(metrics)

    df = pl.DataFrame(records)
    df.write_csv(f'simulations/generated_metrics/{DATASET_NAME}.csv')
