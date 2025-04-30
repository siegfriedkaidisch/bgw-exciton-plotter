import numpy as np
from ase.io import read
from joblib import Parallel, delayed
import time
import os
import sys
from datetime import datetime
from tqdm import tqdm

from library.read_bgw import read_bse_eigenvectors
from library.read_qe import (
    read_wavefunction_q_qe,
    find_min_max_iq_qe,
    get_number_of_valence_bands_qe,
)
from library.misc import (
    direct_to_cart,
    Bohr,
    divide_interval,
    generate_grid,
    write_gaussian_cube,
)

##############################################################################################
####################################        Settings        ##################################
##############################################################################################

# In-files
calc_folder = "/dummy/"
# Path to BGW's eigenvectors.h5
fname_bse_evec = calc_folder + "BGWfolder/" + "eigenvectors.h5"
# Path to folder containing QE wavefunctions (hdf5 and dat supported)
fname_wfn = calc_folder + "QEfolder/QE.save/"
# Path to QE's xml file
fname_e = calc_folder + "QEfolder/QE.xml"
# Path to a file containing the geometry, that is readable by ASE (e.g. QE's input file)
fname_geometry = calc_folder + "QEfolder/QE.in"

# Path where cube-file will be saved to ("_m=xxx_e/h.cube" will be added to fname_out)
fname_out = "density"

# Choose exciton
m = [0,1]  # list of exciton numbers to plot
Q = 0  # Q index, Q-vector must be 0!

# Mode:
# e: hole-averaged electron-density
# h: electron-averaged hole-density
mode = "h"

# Grid settings, number of voxels along unit-cell vectors
factor = 4
n1, n2, n3 = 4*factor, 1*factor, 5*factor

# Decrease this number, if you run out of memory, increase it, to potentially speed up calculations
mem_control = 1.0

##############################################################################################
####################################        Execution        #################################
##############################################################################################

line_splitter = "".join("+-" for _ in range(40)) + "+\n"
print(line_splitter)
line_splitter = "\n" + line_splitter
now = datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))
t1 = time.time()

# Print settings
if mode == "e":
    print(
        f"Calculating hole-averaged electron-density for excitons m={m} and Q_index={Q}."
    )
elif mode == "h":
    print(
        f"Calculating electron-averaged hole-density for excitons m={m} and Q_index={Q}."
    )
print("================= Start of Settings =================")
print("Number of voxels along unit-cell vectors:", n1, n2, n3)
print(f"Memory control set to:", mem_control)
print("=== Input Files ===")
print(f"Path to BGW's BSE eigenvectors (eigenvectors.h5): {fname_bse_evec}")
print(f"Path to QE wavefunctions (WFN folder): {fname_wfn}")
print(f"Path to QE XML file: {fname_e}")
print(f"Path to geometry file (ASE-readable): {fname_geometry}")
print("=== Output ===")
print(f"Output cube file will be saved to: {fname_out}")
print("================== End of Settings ==================")

# Load atoms and generate grid
# load geometry
atoms = read(fname_geometry)
print("Loaded geometry:", atoms.symbols)
# get cell in Angstroem
cell = atoms.cell
print("Unit cell [AA]:")
print(cell.array)
# create grid from (AA)
grid = generate_grid(cell.array, n1, n2, n3)  # (ix,iy,iz, xyz)
print(f"Created real-space grid of shape: {n1}*{n2}*{n3}.")

### Combine mol pieces
# for atom in atoms:
#    if atom.x < -6.0:
#        atom.position += cell[0, :]

# Load BSE eigenvector
bse_evecs, meta_bse = read_bse_eigenvectors(
    fname=fname_bse_evec, n_max=max(m) + 1, meta_only=False, ev_only=True, analyze=False
)
bse_evec = bse_evecs[Q, m]  # remaining: (m, spin, v, c, q)
print(f"Loaded BSE eigenvectors for m={m} and Q_index={Q}.")
print(
    f"Shape: excitons={bse_evec.shape[0]}, spins={bse_evec.shape[1]}, vbands={bse_evec.shape[2]}"
    + f", cbands={bse_evec.shape[3]}, q-pts={bse_evec.shape[4]}"
)
# Check that Q vector is 0:
Q_vector = meta_bse["kpoints"]["exciton_Q_shifts"][Q]
if np.linalg.norm(Q_vector) != 0:
    print("Q-vector must be zero, but it is:", Q_vector)
    sys.exit(1)

# Load some additional needed stuff
# Get number of valence bands of DFT calculation
num_v_dft = get_number_of_valence_bands_qe(fname=fname_e)
print("Number of valence bands in DFT/QE calculation:", num_v_dft)
# Get number of valence and conduction bands used in BSE
num_v_bse = np.shape(bse_evec)[2]
num_c_bse = np.shape(bse_evec)[3]
# Define indices of bands to be loaded
if mode == "e":
    ibands = [num_v_dft + i for i in range(num_c_bse)]
elif mode == "h":
    ibands = sorted([num_v_dft - i - 1 for i in range(num_v_bse)])
print("Wave functions of these bands (count from 0) will be loaded:", ibands)
# Find number of q-points of DFT calculation
i_q_start, i_q_end = find_min_max_iq_qe(wfn_folder=fname_wfn)
print("Number of q-points in DFT/QE calculation:", i_q_end - i_q_start + 1)
# Create one q-list to work on per CPU
njobs = os.cpu_count()
i_q_lists = divide_interval(i_start=i_q_start, i_end=i_q_end, N=njobs)
# Get rec. cell in 1/AA
_, meta = read_wavefunction_q_qe(
    wfn_folder=fname_wfn, iqpt=200, ibands=ibands, meta_only=False
)
rec_cell = meta["b"] / Bohr
print("Reciprocal unit cell [1/AA]:")
print(rec_cell)
# Get spin mode
spin_mode = meta["spin_mode"]
if spin_mode == "su":
    spin_mode_print = "Spin-Unpolarized"
elif spin_mode == "spc":
    spin_mspin_mode_printode = "Spin-Polarized, Collinear"
elif spin_mode == "spn":
    spin_mode_print = "Spin-Polarized, Uncollinear"
print("Detected spin mode:", spin_mode_print)
if spin_mode == "spn":
    print("Not supported.")
    sys.exit(1)

# Load Miller coefficients and wavefunction coefficients for all q-points
print(line_splitter)
now = datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))
print("Starting to load wavefunctions (in momentum-space).")
print("Number of jobs used to load wavefunctions:", njobs)


def load_wfns(q_indices):
    # Prepare output array
    miller_list = []
    c_snq_list = []
    q_list = []
    for iq in q_indices:
        wf, meta = read_wavefunction_q_qe(
            wfn_folder=fname_wfn, iqpt=iq, ibands=ibands, meta_only=False
        )
        miller_list.append(wf["miller"])  # (G, xyz), Miller indices (integers).
        c_snq_list.append(
            wf["cnq"]
        )  # (spin, bands, G), complex wave function coefficients.
        q_list.append(meta["q"])  # (xyz), q-vector in direct coordinates.
    return miller_list, c_snq_list, q_list


tmp = Parallel(n_jobs=njobs, backend="loky", verbose=0)(
    map(delayed(load_wfns), i_q_lists)
)
print("Finished loading wave functions.")
now = datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))
# Untangle tmp into three lists
miller_list = []
c_snq_list = []
q_list = []
for tmp_per_cpu in tmp:
    for miller, c_snq, q in zip(tmp_per_cpu[0], tmp_per_cpu[1], tmp_per_cpu[2]):
        miller_list.append(miller)
        c_snq_list.append(c_snq)
        q_list.append(q)
# miller_list: (q, G, xyz)
# c_snq_list: (q, spin, bands, G)
# q_list: (q,xyz)
# Convert q to numpy array and to cartesian coordinates in 1/AA
q_array = np.array(q_list)
q_array = direct_to_cart(vec=q_array, cell=rec_cell)

# Pre-calculate e^iqr
phaseq = np.exp(1j * np.tensordot(q_array, grid, axes=([1], [3])))  # (iq,ix,iy,iz)
print(
    "Done calculating e^(iqr) for wave function's q-vectors and r being vectors on the real-space grid."
)
print(
    f"Shape: qpts={phaseq.shape[0]}, grid-pts={phaseq.shape[1]}*{phaseq.shape[2]}*{phaseq.shape[3]}"
)

print(line_splitter)
now = datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))
print(
    "Start combining all wave function coefficients into one numpy array. \n"
    + "Since, the Miller-indices are not identical across the q-vectors, \n"
    + "we find the min/max Miller index in each direction and then pad the data."
)

# Get ix/iy/iz range
min_miller = np.min(
    np.array([np.min(miller_list[iq], axis=0) for iq in range(len(miller_list))]),
    axis=0,
)
max_miller = np.max(
    np.array([np.max(miller_list[iq], axis=0) for iq in range(len(miller_list))]),
    axis=0,
)
print("Minimal Miller indices:", min_miller[0], min_miller[1], min_miller[2])
print("Maximal Miller indices:", max_miller[0], max_miller[1], max_miller[2])
x_range_full = np.arange(min_miller[0], max_miller[0] + 1)
y_range_full = np.arange(min_miller[1], max_miller[1] + 1)
z_range_full = np.arange(min_miller[2], max_miller[2] + 1)

# convert c_snq_list to single array on x/y/z meshgird, fill empty spots withh 0.0
nGx, nGy, nGz = len(x_range_full), len(y_range_full), len(z_range_full)
print("Resulting total number of G-vectors:", nGx * nGy * nGz)
c_snq_grids = []
for miller, c_snq in zip(miller_list, c_snq_list):
    nspin, nbands, nG = c_snq.shape
    # Create an empty grid (flattened last three dims) for each c_snq array.
    c_snq_grid = np.zeros((nspin, nbands, nGx * nGy * nGz), dtype=c_snq.dtype)
    # Find the index positions in the x/y/z ranges for each Miller index.
    # np.searchsorted is vectorized and assumes the ranges are sorted.
    x_positions = np.searchsorted(x_range_full, miller[:, 0])
    y_positions = np.searchsorted(y_range_full, miller[:, 1])
    z_positions = np.searchsorted(z_range_full, miller[:, 2])
    # Compute the flat index for each Miller index into the full 3D grid.
    flat_indices = np.ravel_multi_index(
        (x_positions, y_positions, z_positions), (nGx, nGy, nGz)
    )
    # Use advanced indexing to assign all values at once.
    # The flat_indices array selects the corresponding columns in the flattened grid.
    c_snq_grid[:, :, flat_indices] = c_snq  # c_snq has shape (nspin, nbands, nG)
    c_snq_grids.append(c_snq_grid)
# Finally, combine all grids into one array
c_snq = np.array(c_snq_grids)
if mode=="h":
    # Order of bands in c_snq is ascending with band number
    # However, we want the first index to be HOMO, next HOMO-1, etc -> flip axis
    c_snq = np.flip(c_snq, axis=2)
print("Finished combining all wave function coefficients into one numpy array.")
print(
    f"Resulting array is of shape: qpts={c_snq.shape[0]}, spins={c_snq.shape[1]}, "
    + f"bands={c_snq.shape[2]}, G-vectors={c_snq.shape[3]}"
)
now = datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))

# Now, we need to combine c_snq with phaseq and phaseG.
# Shapes:
# c_snq: (q, spin, bands, G)
# phaseq: (q, x,y,z)
# phaseG (coming below): (G, x,y,z)

# Flatten out nGx,nGy,nGz into one index and calculate G vectors
miller_full = np.stack(
    np.meshgrid(x_range_full, y_range_full, z_range_full, indexing="ij"), axis=-1
)
miller_full = np.reshape(miller_full, (-1, 3))  # (nG, 3)
G_full = direct_to_cart(vec=miller_full, cell=rec_cell)  # (nG, 3), G in cart. coords in 1/AA
print("Calculated G-vectors in cartesian coordinates")
print(f"Shape: {G_full.shape[0]}*{G_full.shape[1]}")

# Calculate wave function on real-space grid
print(line_splitter)
now = datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))
print("Starting calculation of wave functions in real space.")
# Create batches of G-indices for vectorized treatment
numG = len(G_full)
numG_batch = int(mem_control * np.floor(2 * 10**9 / (n1 * n2 * n3)))
numG_batch = min(numG_batch, numG)
print(
    "Batch-size (i.e. number of G-vectors to be used in a vectorized fashion)"
    + f" in wave function inverse Fourier transformation: {numG_batch}"
)
G_lists = [
    list(range(i, min(i + numG_batch, numG))) for i in range(0, numG, numG_batch)
]
# Calculate wave function on real-space grid
num_q = c_snq.shape[0]
num_spins = c_snq.shape[1]
num_bands = c_snq.shape[2]
wfn_real_snq = np.zeros((num_q, num_spins, num_bands, n1, n2, n3), dtype="complex128")
print("OPENBLAS_NUM_THREADS:", os.environ.get("OPENBLAS_NUM_THREADS"))
print("MKL_NUM_THREADS:", os.environ.get("MKL_NUM_THREADS"))
for G_list in tqdm(G_lists):
    # Calculate phaseG
    G = G_full[G_list]
    phaseG = np.exp(
        1j * np.tensordot(G, grid, axes=([1], [3]))
    )  # (G, x,y,z) 51% of time
    # Output is huge! -> that's why we split G-vectors up and iterate over batches!

    # Select wavefunction coefs
    c_snq_tmp = c_snq[:, :, :, G_list]  # (q, spin, bands, G)

    # Calculate wavefunction
    tmp = np.tensordot(c_snq_tmp, phaseG, axes=([3], [0]))  # 44% of time
    tmp = tmp * phaseq[:, np.newaxis, np.newaxis, :, :, :]  # 5% of time
    wfn_real_snq += tmp
print("Finished calculation of real-space wave functions.")
now = datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))
print(
    f"Shape: qpts={wfn_real_snq.shape[0]}, spins={wfn_real_snq.shape[1]}, "
    + f"bands={wfn_real_snq.shape[2]}, "
    + f"grid-pts={wfn_real_snq.shape[3]}*{wfn_real_snq.shape[4]}*{wfn_real_snq.shape[5]}"
)

# Prepare BSE eigenvector
if mode == "e":
    # For hole-averaged electron-density, sum over valence bands
    XX = np.einsum("msvcq,msvdq->mscdq", bse_evec, np.conjugate(bse_evec))  # (m,s,c,c',q)
    # XX = np.sum(bse_evec[:,:,:, np.newaxis,:] * np.conjugate(bse_evec[:,:, np.newaxis,:,:]),axis=1)
    print(
        "Multiplied BSE eigenvector with itself, following the formula for the hole-averaged electron-density."
    )
elif mode == "h":
    # For electron-averaged hole-density, sum over conduction bands
    XX = np.einsum("msvcq,mswcq->msvwq", bse_evec, np.conjugate(bse_evec))  # (m,s,v,v',q)
    print(
        "Multiplied BSE eigenvector with itself, following the formula for the electron-averaged hole-density."
    )

print("Finally, putting all parts together and sum over spin, bands and q-vectors.")
if mode == "e":
    # XX: (m,s,c,c',q)
    # wfn_real_snq: (q,s,c,x,y,z)
    XX = np.transpose(XX, (0, 4, 1, 2, 3))  # (m,q,s,c,c')
    XX = XX[:, :, :, :, :, np.newaxis, np.newaxis, np.newaxis]  # (m,q,s,c,c',x,y,z)
    density = wfn_real_snq[np.newaxis, :, :, :, np.newaxis, :, :, :]  # (m,q,s,c, c', x,y,z)
    density = np.sum(XX * density, axis=3)  # (m,q,s, c', x,y,z)
    density = np.sum(density * np.conjugate(wfn_real_snq), axis=3)  # (m,q,s, x,y,z)
    density = np.sum(density, axis=(1, 2))  # (m, x,y,z)
elif mode == "h":
    # XX: (m,s,v,v',q)
    # wfn_real_snq: (q,s,v,x,y,z)
    XX = np.transpose(XX, (0, 4, 1, 2, 3))  # (m, q,s,v,v')
    XX = XX[:, :, :, :, :, np.newaxis, np.newaxis, np.newaxis]  # (m,q,s,v,v',x,y,z)
    density = np.conjugate(
        wfn_real_snq[np.newaxis, :, :, :, np.newaxis, :, :, :]
    )  # (m,q,s,v, v', x,y,z)
    density = np.sum(XX * density, axis=3)  # (m,q,s, v', x,y,z)
    density = np.sum(density * wfn_real_snq, axis=3)  # (m,q,s, x,y,z)
    density = np.sum(density, axis=(1, 2))  # (m, x,y,z)
print(
    f"Done. Shape of density array: excitons={density.shape[0]}, "
    +f"grid={density.shape[1]}*{density.shape[2]}*{density.shape[3]}"
)
dV = cell.volume / (n1 * n2 * n3) # AA^3
density /= np.sum(np.abs(density),axis=(1,2,3))[:,np.newaxis,np.newaxis,np.newaxis] * dV
print("Normalized density to unit cell.")
print("Real part of density:", np.sum(np.abs(np.real(density)),axis=(1,2,3)) * dV)
print(
    "Imaginary part of density (should be zero!):",
    np.sum(np.abs(np.imag(density)),axis=(1,2,3)) * dV,
)
density = np.real(density)
print("Dropped imaginary part.")
print("Minimal value in density array (should not be negative!):", np.min(density,axis=(1,2,3)))

# Create cube file (one per m) and save to disk
if mode == "e":
    line1 = "Hole-averaged electron density;"
elif mode == "h":
    line1 = "Electron-averaged hole density;"
for m_i in m:
    line2 = "Exciton: m-index=" + str(m_i) + ",Q-index=" + str(Q) 
    fname_out_i = fname_out + "_m=" + str(m_i) + "_" + mode + ".cube"
    write_gaussian_cube(
        filename=fname_out_i, atoms=atoms, data=density[m_i], line1=line1, line2=line2
    )
    print(f"Saved density to disk at: {fname_out_i}")

# Print total passed time
print(line_splitter)
now = datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))
t2 = time.time()
print("Finished after " + str((t2 - t1) / 3600) + " hours.", flush=True)
