import numpy as np
import pandas as pd

from CoordMulti import CoordMulti
from Geometry import Geometry


class CountGeom:
    def __init__(
        self,
        geometry: Geometry,
        regions=None,
        use_abs=None,
        wrap_to_180=True,
    ):
        self.origin = geometry
        self.geom_data = geometry.data
        self.kind = geometry.kind
        self.wrap_to_180 = wrap_to_180

        self.value_columns = [c for c in self.geom_data.columns if c != "time"]
        self.n = len(self.value_columns)

        if not self.value_columns:
            raise ValueError("Geometry data has no value columns to count")

        if regions is None:
            # Default for angle-like quantity: cis/trans by absolute value.
            regions = {
                "cis": [(0.0, 90.0, True, False)],
                "trans": [(90.0, np.inf, True, True)],
            }
            if use_abs is None:
                use_abs = True

        if use_abs is None:
            use_abs = False

        self.use_abs = bool(use_abs)
        self.regions = self._normalize_regions(regions)
        self.data = self.count_by_regions()

    @staticmethod
    def _wrap_deg_to_180(values):
        return ((values + 180.0) % 360.0) - 180.0

    @staticmethod
    def _interval_mask(values, low, high, include_low=True, include_high=False):
        if include_low:
            left = values >= low
        else:
            left = values > low

        if np.isinf(high):
            right = np.ones_like(values, dtype=bool)
        elif include_high:
            right = values <= high
        else:
            right = values < high

        return left & right

    @staticmethod
    def _normalize_regions(regions):
        if not isinstance(regions, dict) or not regions:
            raise ValueError(
                "regions must be a non-empty dict, e.g. {'cis': [(0, 90)], 'trans': [(90, np.inf)]}"
            )

        normalized = {}
        for name, spec in regions.items():
            if isinstance(spec, tuple):
                specs = [spec]
            elif isinstance(spec, list):
                specs = spec
            else:
                raise ValueError(
                    f"Invalid region spec for {name}. Use tuple/list of tuples."
                )

            parsed = []
            for one in specs:
                if not isinstance(one, tuple):
                    raise ValueError(
                        f"Invalid interval in region {name}. Expected tuple like (low, high)."
                    )

                if len(one) == 2:
                    low, high = one
                    include_low, include_high = True, False
                elif len(one) == 4:
                    low, high, include_low, include_high = one
                else:
                    raise ValueError(
                        f"Interval in region {name} must have 2 or 4 elements"
                    )

                parsed.append(
                    (
                        float(low),
                        float(high),
                        bool(include_low),
                        bool(include_high),
                    )
                )

            normalized[name] = parsed

        return normalized

    def count_by_regions(self):
        ret = pd.DataFrame({"time": self.geom_data["time"].copy()})

        values = self.geom_data[self.value_columns].to_numpy(dtype=float)

        if self.wrap_to_180 and self.kind in ("angle", "dihedral"):
            values = self._wrap_deg_to_180(values)

        if self.use_abs:
            values = np.abs(values)

        finite = np.isfinite(values)

        for region_name, intervals in self.regions.items():
            mask = np.zeros(values.shape, dtype=bool)
            for low, high, include_low, include_high in intervals:
                mask |= self._interval_mask(
                    values, low, high, include_low, include_high
                )

            mask &= finite
            ret[f"count_{region_name}"] = mask.sum(axis=1)

        return ret

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
    geom = Geometry(coord, "C3", "N2", "N1", "C7")

    count = CountGeom(geom)
    print(count.data)
