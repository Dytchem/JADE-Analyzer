import re

import numpy as np
import pandas as pd
from CoordMulti import CoordMulti


class Geometry:
    def __init__(self, coord_multi: CoordMulti, *atoms):
        if len(atoms) not in (2, 3, 4):
            raise ValueError(
                "Please provide 2 (distance), 3 (angle), or 4 (dihedral) atoms"
            )

        self.coord_multi = coord_multi
        self.atoms = tuple(atoms)
        self.time = coord_multi.data["time"].copy()

        self._traj_indices = self._detect_traj_indices()
        if not self._traj_indices:
            raise ValueError("No trajectory columns detected in coord_multi.data")

        if len(atoms) == 2:
            self.kind = "distance"
        elif len(atoms) == 3:
            self.kind = "angle"
        else:
            self.kind = "dihedral"

        self.data = self._build_dataframe()

    def _detect_traj_indices(self):
        traj_indices = set()
        pattern = re.compile(r"_No\.(\d+)$")
        for col in self.coord_multi.data.columns:
            m = pattern.search(col)
            if m:
                traj_indices.add(int(m.group(1)))
        return sorted(traj_indices)

    def _normalize_atom_label(self, atom_label):
        atom_label = str(atom_label).strip()

        m = re.fullmatch(r"([A-Za-z]+)_?(\d+)", atom_label)
        if m:
            symbol, index = m.group(1), m.group(2)
            return f"{symbol}_{index}"

        m = re.fullmatch(r"([A-Za-z]+)_(\d+)", atom_label)
        if m:
            symbol, index = m.group(1), m.group(2)
            return f"{symbol}_{index}"

        raise ValueError(
            f"Invalid atom label: {atom_label}. Use formats like N1 or N_1"
        )

    def _atom_xyz(self, atom_label, traj_index):
        atom_prefix = self._normalize_atom_label(atom_label)

        x_col = f"{atom_prefix}_x_No.{traj_index}"
        y_col = f"{atom_prefix}_y_No.{traj_index}"
        z_col = f"{atom_prefix}_z_No.{traj_index}"

        missing = [
            c for c in (x_col, y_col, z_col) if c not in self.coord_multi.data.columns
        ]
        if missing:
            raise KeyError(
                f"Missing atom columns for {atom_label} in trajectory No.{traj_index}: {missing}"
            )

        return self.coord_multi.data[[x_col, y_col, z_col]].to_numpy(dtype=float)

    @staticmethod
    def _distance(p1, p2):
        return np.linalg.norm(p2 - p1, axis=1)

    @staticmethod
    def _angle(p1, p2, p3):
        v1 = p1 - p2
        v2 = p3 - p2
        cross_norm = np.linalg.norm(np.cross(v1, v2), axis=1)
        dot_val = np.einsum("ij,ij->i", v1, v2)
        angle_rad = np.arctan2(cross_norm, dot_val)
        return np.degrees(angle_rad)

    @staticmethod
    def _dihedral(p1, p2, p3, p4):
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

    @staticmethod
    def _unwrap_degrees(angle_deg):
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

    def _build_dataframe(self):
        out = pd.DataFrame({"time": self.time})

        atom_text = "-".join(self.atoms)
        for traj_index in self._traj_indices:
            points = [self._atom_xyz(atom, traj_index) for atom in self.atoms]

            if self.kind == "distance":
                value = self._distance(points[0], points[1])
            elif self.kind == "angle":
                value = self._angle(points[0], points[1], points[2])
            else:
                value = self._dihedral(points[0], points[1], points[2], points[3])
                value = self._unwrap_degrees(value)

            out[f"{self.kind}_{atom_text}_No.{traj_index}"] = value

        return out

    def save_to_csv(self, path):
        self.data.to_csv(path, index=False)

    def save_to_pickle(self, path):
        self.data.to_pickle(path)


if __name__ == "__main__":
    path = [
        r"E:\GitHub\JADE-Analyzer\sample\1_del",
        r"E:\GitHub\JADE-Analyzer\sample\1",
        r"E:\GitHub\JADE-Analyzer\sample\2",
    ]
    max_i_time = 500
    coord = CoordMulti(path, max_i_time)
    print(coord.data)

    geom = Geometry(coord, "C3", "N2", "N1", "C7")

    print(geom.data)
