"""
Single trajectory electronic time data handler for JADE-NAMD simulations.

This module provides classes to read and analyze ele_time.out files,
which contain electronic state evolution data including density matrices,
hopping probabilities, and state transitions.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd


class EleTimeSingle:
    """
    Single trajectory electronic time data handler.
    
    Reads and parses ele_time.out files containing electronic state evolution data.
    """

    def __init__(self, path: str, type: str = "file"):
        """
        Initialize the EleTimeSingle instance.
        
        Args:
            path: Path to the ele_time.out file or directory
            type: "file" for direct file path, "folder" for directory containing ele_time.out
        
        Raises:
            FileNotFoundError: If the specified path does not exist
        """
        self.path = Path(path)
        self.type = type
        
        if type == "folder":
            self.file_path = self.path / "ele_time.out"
        else:
            self.file_path = self.path
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"ele_time.out not found at {self.file_path}")
        
        self.data = []
        self._parse_file()
    
    def _parse_complex(self, value_str: str) -> complex:
        """
        Parse a complex number string in the format 'real+iimag'.
        
        Args:
            value_str: String representation of complex number
        
        Returns:
            Parsed complex number
        """
        # Handle format like '0.0000022+i-0.0000000' or '-0.0003584+i 0.0014442'
        # Split by 'i' to separate real and imaginary parts
        parts = value_str.split('i')
        if len(parts) != 2:
            raise ValueError(f"Invalid complex number format: {value_str}")
        
        # Remove trailing '+' from real part if present
        real_str = parts[0].strip()
        if real_str.endswith('+'):
            real_str = real_str[:-1]
        
        real_part = float(real_str)
        
        # Handle empty or whitespace-only imaginary part (treat as 0)
        imag_str = parts[1].strip()
        if not imag_str:
            imag_part = 0.0
        else:
            imag_part = float(imag_str)
        
        return complex(real_part, imag_part)
    
    def _parse_block(self, block_lines: list) -> dict:
        """
        Parse a single data block (between separator lines).
        
        Args:
            block_lines: List of lines in the block
        
        Returns:
            Dictionary containing parsed data for this time step
        """
        data = {
            'step': None,
            'time': None,
            'rho': np.zeros((2, 2), dtype=complex),
            'current_state': None,
            'hopping_prob': None,
            'area_for_hopping': None,
            'hopping_prob_avg': None,
            'random_number': None,
            'new_state': None
        }
        
        for line in block_lines:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 3:
                continue
            
            step = int(parts[0])
            time = float(parts[1])
            
            if data['step'] is None:
                data['step'] = step
                data['time'] = time
            
            if len(parts) >= 6 and parts[2] == 'rho':
                row = int(parts[3]) - 1
                col = int(parts[4]) - 1
                if 0 <= row < 2 and 0 <= col < 2:
                    data['rho'][row, col] = self._parse_complex(parts[5])
            
            elif len(parts) >= 5 and parts[2] == 'The' and parts[3] == 'current' and parts[4] == 'state':
                data['current_state'] = int(parts[5])
            
            elif len(parts) >= 6 and parts[2] == 'Hopping' and parts[3] == 'probability' and parts[4] == '(averaged)':
                data['hopping_prob_avg'] = (float(parts[5]), float(parts[6]))
            
            elif len(parts) >= 6 and parts[2] == 'Hopping' and parts[3] == 'probability':
                # Only match if not followed by (averaged)
                if len(parts) >= 5 and parts[4] != '(averaged)':
                    data['hopping_prob'] = (float(parts[4]), float(parts[5]))
            
            elif len(parts) >= 7 and parts[2] == 'Area' and parts[3] == 'for' and parts[4] == 'hopping':
                data['area_for_hopping'] = (float(parts[5]), float(parts[6]))
            
            elif len(parts) >= 4 and parts[2] == 'Random' and parts[3] == 'number':
                data['random_number'] = float(parts[4])
            
            elif len(parts) >= 5 and parts[2] == 'The' and parts[3] == 'new' and parts[4] == 'state':
                data['new_state'] = int(parts[5])
        
        return data
    
    def _parse_file(self):
        """Parse the ele_time.out file and store data in self.data."""
        with open(self.file_path, 'r') as f:
            content = f.read()
        
        # Split by separator lines
        blocks = re.split(r'-{36,}', content)
        
        for block in blocks:
            lines = block.strip().split('\n')
            if lines and lines[0]:
                parsed = self._parse_block(lines)
                if parsed['step'] is not None:
                    self.data.append(parsed)
    
    def get_steps(self) -> list:
        """Get all step numbers."""
        return [d['step'] for d in self.data]
    
    def get_times(self) -> np.ndarray:
        """Get all time values as numpy array."""
        return np.array([d['time'] for d in self.data])
    
    def get_rho_matrix(self, step: int = None) -> np.ndarray:
        """
        Get density matrix at specific step or all steps.
        
        Args:
            step: Step number to get density matrix for. If None, returns all.
        
        Returns:
            Density matrix (2x2) or array of density matrices (Nx2x2)
        """
        if step is not None:
            for d in self.data:
                if d['step'] == step:
                    return d['rho']
            raise ValueError(f"Step {step} not found")
        
        return np.array([d['rho'] for d in self.data])
    
    def get_populations(self) -> np.ndarray:
        """
        Get electronic state populations over time.
        
        Returns:
            Array of shape (n_steps, 2) containing population probabilities
        """
        populations = []
        for d in self.data:
            pop1 = np.real(d['rho'][0, 0])
            pop2 = np.real(d['rho'][1, 1])
            populations.append([pop1, pop2])
        return np.array(populations)
    
    def get_states(self) -> list:
        """Get current state at each step."""
        return [d['current_state'] for d in self.data]
    
    def get_new_states(self) -> list:
        """Get new state after hopping at each step."""
        return [d['new_state'] for d in self.data]
    
    def get_hopping_probabilities(self) -> np.ndarray:
        """Get hopping probabilities as array."""
        probs = []
        for d in self.data:
            if d['hopping_prob'] is not None:
                probs.append(list(d['hopping_prob']))
            else:
                probs.append([np.nan, np.nan])
        return np.array(probs)
    
    def get_random_numbers(self) -> np.ndarray:
        """Get random numbers as array."""
        return np.array([d['random_number'] for d in self.data])
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert data to pandas DataFrame."""
        rows = []
        for d in self.data:
            row = {
                'step': d['step'],
                'time': d['time'],
                'rho_11_real': np.real(d['rho'][0, 0]),
                'rho_11_imag': np.imag(d['rho'][0, 0]),
                'rho_12_real': np.real(d['rho'][0, 1]),
                'rho_12_imag': np.imag(d['rho'][0, 1]),
                'rho_21_real': np.real(d['rho'][1, 0]),
                'rho_21_imag': np.imag(d['rho'][1, 0]),
                'rho_22_real': np.real(d['rho'][1, 1]),
                'rho_22_imag': np.imag(d['rho'][1, 1]),
                'current_state': d['current_state'],
                'new_state': d['new_state'],
                'random_number': d['random_number']
            }
            if d['hopping_prob'] is not None:
                row['hop_prob_1'], row['hop_prob_2'] = d['hopping_prob']
            if d['area_for_hopping'] is not None:
                row['area_1'], row['area_2'] = d['area_for_hopping']
            if d['hopping_prob_avg'] is not None:
                row['hop_prob_avg_1'], row['hop_prob_avg_2'] = d['hopping_prob_avg']
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def save_to_csv(self, output_path: str):
        """Save data to CSV file."""
        df = self.to_dataframe()
        df.to_csv(output_path, index=False)
    
    def description(self) -> dict:
        """Get summary description of the data."""
        return {
            'n_steps': len(self.data),
            'time_range': (self.data[0]['time'], self.data[-1]['time']),
            'initial_state': self.data[0]['current_state'],
            'final_state': self.data[-1]['new_state'],
            'n_hops': sum(1 for d in self.data if d['current_state'] != d['new_state'])
        }
