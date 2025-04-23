# BerkeleyGW Exciton Plotter

A Python toolkit for computing exciton electron- or hole-density distributions (hole-averaged electron-density or electron-averaged hole-density) from BerkeleyGW Bethe–Salpeter Equation (BSE) eigenvectors and Quantum ESPRESSO (QE) wavefunctions. Outputs Gaussian cube files suitable for visualization in VESTA, VMD, or other visualization software.

---

## Features

- **Arbitrary Unit Cells**: Supports non-rectangular (triclinic, monoclinic, etc.) unit cells.
- **Spin Treatment**: Handles spin-unpolarized (SU) and spin-polarized collinear (SPC) calculations.
- **Exciton Types**: Only spin-singlet excitons are supported.
- **Momentum Transfer**: Only zero momentum transfer (`Q = 0`) excitons.
- **Flexible Grid**: Customizable real-space grid resolution `(n₁ × n₂ × n₃)`.
- **Memory Control**: Adjustable batching for Fourier transforms to suit available memory.

---

## Requirements

- **Python 3.10+**
- **Packages**:
  - `numpy`
  - `ase`
  - `joblib`
  - `tqdm`
  - `h5py`

- **Data**:
  - BerkeleyGW BSE eigenvectors file (`eigenvectors.h5`)
  - Quantum ESPRESSO save directory (containing wave functions as `hdf5` or `dat` files) and `XML` output file
  - Geometry file readable by ASE (e.g., QE input file)

---

## Installation

```bash
git clone https://github.com/siegfriedkaidisch/bgw-exciton-plotter.git

pip install numpy ase joblib tqdm h5py
```

---

## Usage

Edit the script parameters at the top of `plotdens.py`:

```python
# Input paths
fname_bse_evec = "/path/to/eigenvectors.h5"
fname_wfn      = "/path/to/qe.save/"
fname_e        = "/path/to/qe.xml"
fname_geometry = "/path/to/geometry"

# Which density to plot
m    = 0      # exciton index
Q    = 0      # Q-index (such that Q-vector is 0)
mode = "h"    # "h" for hole density, "e" for electron density

# Grid resolution
n1, n2, n3 = 16, 4, 20

# Memory control for FFT batching
mem_control = 1.0

# Output
fname_out = "density.cube"
```

Then run:

```bash
cd bgw-exciton-plotter

python plotdens.py
```

The script prints progress and writes `density.cube`, which can be loaded in visualization tools.

---

## Current Limitations

- **No Noncollinear Spin**: Only spin-unpolarized (SU) and spin-polarized collinear (SPC) wavefunctions are supported.
- **Zero Momentum Transfer Only**: `Q ≠ 0` excitons are not implemented.
- **Spin-Singlet Only**: Triplet excitons are not supported.

---

## Authors

Siegfried Kaidisch (siegfried.kaidisch(at)uni-graz.at)

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## Known issues

For a list of known issues please see the [issues page on GitHub](https://github.com/siegfriedkaidisch/bgw-exciton-plotter/issues), otherwise please open a new issue.
