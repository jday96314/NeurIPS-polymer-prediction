from typing import Any, Optional, Tuple, Dict

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import msd, polymer
from MDAnalysis.lib import mdamath
import joblib
from rdkit.Chem import Descriptors3D
from tqdm import tqdm
import math
from MDAnalysis.exceptions import NoDataError
import glob
import polars as pl
from multiprocessing import Pool
import traceback
from typing import Callable, Dict
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdMD

# -----------------------------
# FFV (Monte Carlo, freud)
# -----------------------------


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


def compute_all_3d_descriptors(molecule: Chem.Mol, molecule_name: str) -> Dict[str, float]:
    """
    Compute a broad set of 3D descriptors that RDKit provides for a molecule with a precomputed conformer.

    Keys are formatted as "{molecule_name}_{descriptor_name}".
    Safely skips any descriptor that is unavailable in the local RDKit build or raises at runtime.

    Core descriptors attempted (if present in your RDKit):
      - Asphericity
      - Eccentricity
      - InertialShapeFactor
      - NPR1, NPR2
      - PMI1, PMI2, PMI3
      - RadiusOfGyration
      - SpherocityIndex
      - PBF  (Plane of Best Fit; some builds only)
      - LabuteASA (approximate ASA; does not require rdFreeSASA)

    Optional extras (if your RDKit is compiled with them):
      - FreeSASA (total solvent-accessible surface area via rdFreeSASA)

    Parameters
    ----------
    molecule : rdkit.Chem.Mol
        Molecule with at least one 3D conformer already embedded (e.g., via ETKDG + UFF/MMFF).
    molecule_name : str
        Name used as a prefix for descriptor keys.

    Returns
    -------
    Dict[str, float]
        Mapping from "{molecule_name}_{descriptor}" to the computed float value.
    """
    if molecule is None:
        raise ValueError("molecule must be a valid RDKit Mol.")
    if molecule.GetNumConformers() == 0:
        raise ValueError("molecule must have at least one precomputed 3D conformer.")

    # Use the first conformer by default; many RDKit descriptor functions accept confId, others do not.
    conformer_id: int = 0

    def _call_descriptor(descriptor_func: Callable[[Chem.Mol], float], *, name_hint: str) -> float | None:
        """
        Call a descriptor function, passing confId if supported; return None on failure.
        """
        try:
            # Try passing confId= for functions that support it.
            return float(descriptor_func(molecule, confId=conformer_id))  # type: ignore[arg-type]
        except TypeError:
            # Fallback for functions without confId.
            try:
                return float(descriptor_func(molecule))
            except Exception:
                traceback.print_exc()
                return None
        except Exception:
            traceback.print_exc()
            return None

    descriptor_functions: dict[str, Callable[..., float]] = {}

    # Core 3D shape descriptors provided by rdMolDescriptors (conditionally added if present).
    for descriptor_name in [
        "CalcAsphericity",
        "CalcEccentricity",
        "CalcInertialShapeFactor",
        "CalcNPR1",
        "CalcNPR2",
        "CalcPMI1",
        "CalcPMI2",
        "CalcPMI3",
        "CalcRadiusOfGyration",
        "CalcSpherocityIndex",
        "CalcPBF",           # Optional: only on newer/complete builds
        "CalcLabuteASA",     # Approximate ASA (not the same as FreeSASA)
    ]:
        descriptor_func = getattr(rdMD, descriptor_name, None)
        if callable(descriptor_func):
            # Store with a clean, user-facing descriptor key (strip "Calc" prefix).
            clean_name = descriptor_name.removeprefix("Calc")
            descriptor_functions[clean_name] = descriptor_func  # type: ignore[assignment]

    descriptor_values: Dict[str, float] = {}

    for clean_name, descriptor_func in descriptor_functions.items():
        value = _call_descriptor(descriptor_func, name_hint=clean_name)
        if value is not None:
            key = f"{molecule_name}_{clean_name}"
            descriptor_values[key] = value

    return descriptor_values


# -----------------------------
# High-level API
# -----------------------------

def compute_polymer_metrics(results_directory_path: str) -> dict[str, Any]:
    # LOAD SIMULATION RESULTS.
    universe = mda.Universe(
        f'{results_directory_path}/eq1_final.data',
        f'{results_directory_path}/eq1_short.xtc',
    )

    last_frame = universe.trajectory[-2]

    # COMPUTE DENSITY.
    amu_to_g = 1.66053906660e-24
    volume_angstrom3 = mdamath.box_volume(universe.dimensions)
    rho_g_cm3 = (universe.atoms.masses.sum() * amu_to_g) / (volume_angstrom3 * 1e-24)

    # COMPUTE FFV.
    ffv_stats = compute_fractional_free_volume(universe=universe)

    # COMPUTE Rg + SHAPE METRICS.
    gyr_eigs = mass_weighted_gyration_eigenvalues(universe.atoms)
    shape = shape_metrics_from_eigs(gyr_eigs)
    rg_global_angstrom = float(np.sqrt(np.sum(gyr_eigs)))

    # COMPUTE DIFFUSIVITY.
    D_A2ps, D_cm2s = compute_diffusivity(universe, selection="all", time_per_frame_ps=None)

    # COMPUTE PERSISTENCE LENGTH STATS.
    backbone_chains = build_backbone_paths_heavy_atoms(universe, min_atoms_per_chain=12)
    number_of_backbone_chains = len(backbone_chains)
    if number_of_backbone_chains > 0:
        persistence_stats = compute_persistence_length_stats(universe, backbone_chains)

    # COMPUTE 3D DESCRIPTORS.
    monomer = joblib.load(f'{results_directory_path}/mol.pkl')
    monomer_descriptors = compute_all_3d_descriptors(monomer, molecule_name='monomer')

    homopoly = joblib.load(f'{results_directory_path}/homopoly.pkl')
    homopoly_descriptors = compute_all_3d_descriptors(homopoly, molecule_name='homopoly')

    return {
        "density_g_cm3": float(rho_g_cm3),
        **ffv_stats,
        "rg_angstrom": float(rg_global_angstrom),
        **shape,
        **persistence_stats,
        "diffusivity_A2_per_ps": float(D_A2ps),
        "diffusivity_cm2_per_s": float(D_cm2s),
        "n_backbone_chains": int(number_of_backbone_chains),
        **monomer_descriptors,
        **homopoly_descriptors,
    }


if __name__ == "__main__":
    # DATASET_NAME = 'host_hybrid'
    # all_smiles = pl.read_csv('data/from_host/train.csv')['SMILES'].to_list()

    DATASET_NAME = 'PI1M_hybrid'
    all_smiles = pl.read_csv('data/PI1M/PI1M.csv')['SMILES'].to_list()

    final_dump_filepaths = glob.glob(f'simulations/work_dirs/{DATASET_NAME}/*/eq1_final.dump')
    data_directory_paths = [path.replace('/eq1_final.dump', '') for path in final_dump_filepaths]

    # data_directory_paths = data_directory_paths[:5]
    with Pool(processes=16) as pool:
        raw_records = pool.map(compute_polymer_metrics, data_directory_paths)

    records = []
    for data_directory_path, metrics in zip(data_directory_paths, raw_records):
        smiles = all_smiles[int(data_directory_path.split('/')[-1])]
        record = {'SMILES': smiles}
        record.update(metrics)
        records.append(record)

    df = pl.DataFrame(records)
    df.write_csv(f'simulations/generated_metrics/{DATASET_NAME}_09_13.csv')
