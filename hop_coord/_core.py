import re

import numpy as np


def normalize_atom_label(atom_label):
    atom_label = str(atom_label).strip()

    match = re.fullmatch(r"([A-Za-z]+)_?(\d+)", atom_label)
    if match:
        symbol, index = match.group(1), match.group(2)
        return f"{symbol}_{index}"

    raise ValueError(f"Invalid atom label: {atom_label}. Use formats like N1 or N_1")


def geometry_kind(atom_count):
    if atom_count == 2:
        return "distance"
    if atom_count == 3:
        return "angle"
    if atom_count == 4:
        return "dihedral"
    raise ValueError("Please provide 2 (distance), 3 (angle), or 4 (dihedral) atoms")


def geometry_label(atom_count, atoms):
    return f"{geometry_kind(atom_count)}_{'-'.join(atoms)}"


def normalize_geometries(*geometries):
    # Backward compatible: HopCoord*(..., "A1", "B2", "C3") means one geometry.
    if not geometries:
        raise ValueError("Please provide at least one geometry definition")

    first = geometries[0]
    if isinstance(first, str):
        if len(geometries) not in (2, 3, 4):
            raise ValueError(
                "Single geometry must have 2, 3, or 4 atom labels, e.g. 'N1', 'N2', 'N3'"
            )
        return [tuple(str(x) for x in geometries)]

    normalized = []
    for geom in geometries:
        if not isinstance(geom, (list, tuple)):
            raise ValueError(
                "Each geometry must be a list/tuple of atom labels, e.g. ('N1','N2','N3')"
            )
        if len(geom) not in (2, 3, 4):
            raise ValueError(f"Geometry {geom} must contain 2, 3, or 4 atom labels")
        normalized.append(tuple(str(x) for x in geom))

    return normalized


def atom_xyz(coord_df, atom_label, traj_index=None):
    atom_prefix = normalize_atom_label(atom_label)
    suffix = f"_No.{traj_index}" if traj_index is not None else ""

    x_col = f"{atom_prefix}_x{suffix}"
    y_col = f"{atom_prefix}_y{suffix}"
    z_col = f"{atom_prefix}_z{suffix}"

    missing = [c for c in (x_col, y_col, z_col) if c not in coord_df.columns]
    if missing:
        if traj_index is None:
            raise KeyError(f"Missing atom columns for {atom_label}: {missing}")
        raise KeyError(
            f"Missing atom columns for {atom_label} in trajectory No.{traj_index}: {missing}"
        )

    return coord_df[[x_col, y_col, z_col]].to_numpy(dtype=float)


def distance(p1, p2):
    return np.linalg.norm(p2 - p1, axis=1)


def angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    cross_norm = np.linalg.norm(np.cross(v1, v2), axis=1)
    dot_val = np.einsum("ij,ij->i", v1, v2)
    angle_rad = np.arctan2(cross_norm, dot_val)
    return np.degrees(angle_rad)


def dihedral(p1, p2, p3, p4):
    ab = p2 - p1
    bc = p3 - p2
    cd = p4 - p3

    ab_norm = np.linalg.norm(ab, axis=1, keepdims=True)
    bc_norm = np.linalg.norm(bc, axis=1, keepdims=True)
    cd_norm = np.linalg.norm(cd, axis=1, keepdims=True)

    with np.errstate(invalid="ignore", divide="ignore"):
        ab_unit = ab / ab_norm
        bc_unit = bc / bc_norm
        cd_unit = cd / cd_norm

    m = np.cross(ab_unit, bc_unit)
    n = np.cross(bc_unit, cd_unit)

    m_norm = np.linalg.norm(m, axis=1, keepdims=True)
    n_norm = np.linalg.norm(n, axis=1, keepdims=True)

    with np.errstate(invalid="ignore", divide="ignore"):
        m_unit = m / m_norm
        n_unit = n / n_norm

    x = np.einsum("ij,ij->i", m_unit, n_unit)
    y = np.einsum("ij,ij->i", np.cross(m_unit, n_unit), bc_unit)

    angle_rad = np.arctan2(y, x)
    return np.degrees(angle_rad)


def unwrap_degrees(angle_deg):
    angle_deg = np.asarray(angle_deg, dtype=float)
    unwrapped = angle_deg.copy()

    finite_mask = np.isfinite(angle_deg)
    if not finite_mask.any():
        return unwrapped

    valid_idx = np.where(finite_mask)[0]
    start = 0
    while start < len(valid_idx):
        end = start
        while end + 1 < len(valid_idx) and valid_idx[end + 1] == valid_idx[end] + 1:
            end += 1

        seg = valid_idx[start : end + 1]
        rad = np.deg2rad(angle_deg[seg])
        unwrapped[seg] = np.rad2deg(np.unwrap(rad, discont=np.pi))
        start = end + 1

    return unwrapped


def rewrap_degrees(angle_deg):
    angle_deg = np.asarray(angle_deg, dtype=float)
    wrapped = angle_deg.copy()

    finite_mask = np.isfinite(angle_deg)
    if finite_mask.any():
        wrapped[finite_mask] = ((angle_deg[finite_mask] + 180.0) % 360.0) - 180.0

    return wrapped


def build_geometry_series(coord_df, atoms, traj_index=None, unwrap_dihedral=True):
    label = geometry_label(len(atoms), atoms)
    kind = geometry_kind(len(atoms))

    def _postprocess_precomputed(values):
        values = np.asarray(values, dtype=float)
        if kind != "dihedral":
            return values
        if unwrap_dihedral:
            return unwrap_degrees(values)
        return rewrap_degrees(values)

    if traj_index is not None:
        col = f"{label}_No.{traj_index}"
        if col in coord_df.columns:
            return _postprocess_precomputed(coord_df[col].to_numpy(dtype=float))
    else:
        if label in coord_df.columns:
            return _postprocess_precomputed(coord_df[label].to_numpy(dtype=float))
        col_no1 = f"{label}_No.1"
        if col_no1 in coord_df.columns:
            return _postprocess_precomputed(coord_df[col_no1].to_numpy(dtype=float))

    points = [atom_xyz(coord_df, atom, traj_index=traj_index) for atom in atoms]

    if len(atoms) == 2:
        return distance(points[0], points[1])
    if len(atoms) == 3:
        return angle(points[0], points[1], points[2])

    value = dihedral(points[0], points[1], points[2], points[3])
    if unwrap_dihedral:
        return unwrap_degrees(value)
    return value


def extract_hop_time(state_values, time_values, time_interval):
    state_values = np.asarray(state_values, dtype=int)
    time_values = np.asarray(time_values, dtype=float)

    if state_values.size == 0 or time_values.size == 0:
        return np.nan, "empty"

    last_idx = len(state_values) - 1

    if state_values[last_idx] == 2:
        return float(time_values[last_idx] + time_interval), "no_hop"

    if state_values[last_idx] == 1:
        for j in range(last_idx - 1, -1, -1):
            if state_values[j] == 2:
                return float(time_values[j + 1]), "hop"
        return np.nan, "no_hop"

    flag = False
    for j in range(last_idx - 1, -1, -1):
        if state_values[j] == 2:
            if flag:
                return float(time_values[j + 1]), "hop"
            return np.nan, "crash"
        if state_values[j] == 1:
            if not flag:
                return np.nan, "crash"
            flag = True

    return np.nan, "crash"


def select_value_at_time(time_values, series_values, target_time, atol=1e-8):
    time_values = np.asarray(time_values, dtype=float)
    series_values = np.asarray(series_values, dtype=float)

    if not np.isfinite(target_time):
        return np.nan, np.nan, np.nan, "missing"

    matches = np.where(np.isclose(time_values, target_time, atol=atol, rtol=1e-7))[0]
    if len(matches) > 0:
        idx = int(matches[0])
        return (
            float(series_values[idx]),
            float(time_values[idx]),
            idx,
            "matched",
        )

    return np.nan, np.nan, np.nan, "out_of_range"
