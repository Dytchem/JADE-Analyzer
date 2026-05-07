import os

import numpy as np
import pandas as pd

from unite.base import BaseData


class EnergySingle(BaseData):
    def __init__(self, path, max_i_time, type="folder"):
        self.max_i_time = max_i_time
        
        if type == "folder":
            data = self._parse_energy_time(path)
        
        elif type == "csv":
            data = pd.read_csv(path)
        
        elif type == "pickle":
            data = pd.read_pickle(path)
        
        else:
            raise ValueError("type must be 'folder' or 'csv' or 'pickle'")
        
        super().__init__(data, max_i_time, "energy")

    def _parse_energy_time(self, path):
        energy_path = os.path.join(path, "energy_time.out")
        
        if not os.path.exists(energy_path):
            raise FileNotFoundError(f"energy_time.out not found in {path}")
        
        with open(energy_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        columns = ["step", "time", "pot_ene", "kin_ene", "tot_ene", "lin_mom_x", "lin_mom_y", "lin_mom_z"]
        data = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            parts = line.split()
            if len(parts) >= 8:
                row = [
                    int(parts[0]),
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                    float(parts[5]),
                    float(parts[6]),
                    float(parts[7]),
                ]
                data.append(row)
        
        df = pd.DataFrame(data, columns=columns)
        
        frame_count = self.max_i_time + 1
        if len(df) < frame_count:
            padding = pd.DataFrame(
                np.nan,
                index=np.arange(len(df), frame_count),
                columns=columns
            )
            df = pd.concat([df, padding], ignore_index=True)
        
        df = df.iloc[:frame_count]
        
        return df

    def save_to_csv(self, path):
        self.data.to_csv(path, index=False)

    def save_to_pickle(self, path):
        self.data.to_pickle(path)


if __name__ == "__main__":
    path = r"E:\GitHub\JADE-Analyzer\sample\1"
    max_i_time = 500
    energy = EnergySingle(path, max_i_time)
    energy.save_to_csv(r"E:\GitHub\JADE-Analyzer\output\energy_single.csv")

    print(energy)
    print(energy.data.head())
