import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))


class DiSingle:
    def __init__(self, path, max_i_time, type="folder"):
        self.max_i_time = max_i_time
        if type == "folder":
            mulliken_frames, dipole_frames, time_frames = self._parse_di_time(path)
            self.data = self._build_dataframe(mulliken_frames, dipole_frames, time_frames)
        elif type == "csv":
            self.data = pd.read_csv(path)
        elif type == "pickle":
            self.data = pd.read_pickle(path)
        else:
            raise ValueError("type must be 'folder' or 'csv' or 'pickle'")
        self.time_interval = self.data["time"][1] if len(self.data) > 1 else np.nan

    def set_time_series(self, time_series):
        time_array = np.asarray(time_series, dtype=float)
        if time_array.ndim != 1:
            raise ValueError("time_series must be a 1D sequence")
        if len(time_array) != len(self.data):
            raise ValueError(
                f"time_series length {len(time_array)} does not match data length {len(self.data)}"
            )

        self.data.loc[:, "time"] = time_array
        self.time_interval = (
            time_array[1] - time_array[0] if len(time_array) > 1 else np.nan
        )

    def _parse_di_time(self, path):
        mulliken_frames = []
        dipole_frames = []
        time_frames = []
        
        di_path = os.path.join(path, "di_time.out")
        
        with open(di_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        i = 0
        num_atoms = None
        
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith("STEP"):
                frame_mulliken = {}
                frame_dipole = {}
                frame_time = np.nan
                i += 1
                
                while i < len(lines):
                    current_line = lines[i].strip()
                    
                    if current_line.startswith("i_time ="):
                        frame_time = float(current_line.split("=")[1].strip())
                        i += 1
                    
                    elif current_line.startswith("Atom") and "Mulliken Charge" in current_line:
                        i += 1
                        while i < len(lines):
                            state_line = lines[i].strip()
                            if state_line.startswith("State"):
                                state_label = state_line.split()[1]
                                i += 1
                                atom_idx = 1
                                while i < len(lines):
                                    atom_line = lines[i].strip()
                                    if not atom_line or atom_line.startswith("State") or atom_line.startswith("Dipole"):
                                        break
                                    parts = atom_line.split()
                                    if len(parts) >= 2:
                                        charge = float(parts[-1])
                                        frame_mulliken[f"mulliken_{state_label}_atom{atom_idx}"] = charge
                                        atom_idx += 1
                                    i += 1
                                if num_atoms is None:
                                    num_atoms = atom_idx - 1
                            elif state_line.startswith("Dipole"):
                                break
                            else:
                                i += 1
                    
                    elif current_line.startswith("Dipole"):
                        i += 1
                        while i < len(lines):
                            dipole_line = lines[i].strip()
                            if not dipole_line or dipole_line.startswith("---"):
                                break
                            parts = dipole_line.split()
                            if len(parts) >= 3:
                                operator = parts[0]
                                au_value = float(parts[1])
                                debye_value = float(parts[2])
                                frame_dipole[f"dipole_{operator}_au"] = au_value
                                frame_dipole[f"dipole_{operator}_debye"] = debye_value
                            i += 1
                    
                    elif line.startswith("---") or current_line == "":
                        i += 1
                    
                    elif current_line.startswith("STEP"):
                        break
                    
                    else:
                        i += 1
                
                mulliken_frames.append(frame_mulliken)
                dipole_frames.append(frame_dipole)
                time_frames.append(frame_time)
            
            else:
                i += 1
        
        return mulliken_frames, dipole_frames, time_frames

    def _build_dataframe(self, mulliken_frames, dipole_frames, time_frames):
        if not mulliken_frames or not dipole_frames:
            return pd.DataFrame()
        
        frame_count = self.max_i_time + 1
        all_columns = set()
        
        for mf, df in zip(mulliken_frames, dipole_frames):
            all_columns.update(mf.keys())
            all_columns.update(df.keys())
        
        all_columns = sorted(list(all_columns))
        values = np.full((frame_count, len(all_columns)), np.nan)
        
        valid_frames = min(len(mulliken_frames), frame_count)
        for frame_idx in range(valid_frames):
            mf = mulliken_frames[frame_idx]
            df = dipole_frames[frame_idx]
            for col_idx, col in enumerate(all_columns):
                if col in mf:
                    values[frame_idx, col_idx] = mf[col]
                elif col in df:
                    values[frame_idx, col_idx] = df[col]
        
        if len(time_frames) > 1 and not np.isnan(time_frames[1]) and time_frames[1] > 0:
            time = np.arange(0, time_frames[1] * (self.max_i_time + 0.5), time_frames[1])
        else:
            time = np.arange(0, frame_count, 1.0)
        
        df = pd.DataFrame(values, columns=all_columns)
        df.insert(0, "time", time)
        return df

    def save_to_csv(self, path):
        self.data.to_csv(path, index=False)

    def save_to_pickle(self, path):
        self.data.to_pickle(path)


if __name__ == "__main__":
    path = r"E:\GitHub\JADE-Analyzer\sample\1_del"
    max_i_time = 500
    di = DiSingle(path, max_i_time)
    di.save_to_csv(r"E:\GitHub\JADE-Analyzer\output\di_single.csv")

    print(di.data)
