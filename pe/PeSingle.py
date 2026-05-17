import os

import pandas as pd

from unite.base import BaseData


class PeSingle(BaseData):
    def __init__(self, path, max_i_time, type="folder"):
        self.max_i_time = max_i_time

        if type == "folder":
            data = self._parse_pe_time(path)
        elif type == "csv":
            data = pd.read_csv(path)
        elif type == "pickle":
            data = pd.read_pickle(path)
        else:
            raise ValueError("type must be 'folder' or 'csv' or 'pickle'")

        super().__init__(data, max_i_time, "pe")

    def _parse_pe_time(self, path):
        pe_path = os.path.join(path, "pe_time.out")

        if not os.path.exists(pe_path):
            raise FileNotFoundError(f"pe_time.out not found in {path}")

        columns = ["step", "time", "e0", "e1", "e_now"]
        data = []

        with open(pe_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                if len(parts) < 5:
                    continue

                try:
                    row = [
                        int(parts[0]),
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3]),
                        float(parts[4]),
                    ]
                except ValueError:
                    continue

                data.append(row)

        df = pd.DataFrame(data, columns=columns)
        return df

    def save_to_csv(self, path):
        self.data.to_csv(path, index=False)

    def save_to_pickle(self, path):
        self.data.to_pickle(path)


if __name__ == "__main__":
    path = r"E:\GitHub\JADE-Analyzer\sample\1"
    max_i_time = 500
    pe = PeSingle(path, max_i_time)
    pe.save_to_csv(r"E:\GitHub\JADE-Analyzer\output\pe_single.csv")

    print(pe)
    print(pe.data.head())
