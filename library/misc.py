import numpy as np

Hartree = 27.211386024367243  # 1 Hartree = 27.2 eV
Bohr = 0.5291772105638411  # 1 Bohr = 0.5 Angstroem


def divide_interval(i_start, i_end, N):
    """
    Distribute numbers in the interval [i_start, i_end] across N lists in a round-robin fashion.

    Parameters
    ----------
    i_start : int
        Start of the main interval.
    i_end : int
        End of the main interval.
    N : int
        Number of lists to create.

    Returns
    -------
    list of lists
        Each list contains integers spaced apart by N, starting from i_start to i_end.

    Notes
    -----
    This function spreads the numbers from i_start to i_end across N lists such that each list
    contains numbers that are distributed in a round-robin manner. This is similar to dealing cards
    into N hands.

    Examples
    --------
    >>> divide_interval(1, 10, 3)
    [[1, 4, 7, 10], [2, 5, 8], [3, 6, 9]]

    """
    # Create N empty lists to hold the results
    result_lists = [[] for _ in range(N)]

    # Distribute the numbers in a round-robin fashion
    for number in range(i_start, i_end + 1):
        # Find the appropriate list to place this number
        list_index = (number - i_start) % N
        result_lists[list_index].append(number)

    return result_lists


def direct_to_cart(vec, cell, transpose=True, npa=True):
    """
    Convert vectors from direct to Cartesian coordinates.

    This function converts a set of vectors from direct (fractional) coordinates to
    Cartesian coordinates, using the provided (reciprocal) lattice cell vectors. It supports
    conversion of multiple vectors at once and allows for the lattice cell vectors to be input
    either as row vectors (default) or as column vectors.
    Ad Conversion: coords_cart = coords_frac_1 * lattice_vec_1 + ... (same for direction 2 and 3)
    This is the same as a matrix-vector multiplication, with the matrix's columns being
    the lattice vectors and the (column) vector containing the fractional coordinates.

    Note
    ----
    The returned vectors will be in the same units as the cell vectors.

    Parameters
    ----------
    vec : array_like
        The vectors to be converted, in direct coordinates. This can be a 1D array
        for a single vector (numpy.ndarray of shape (3,) or a list of length 3)
        or a 2D array, where each row represents a different vector and the last dimension
        corresponds to the spatial directions (numpy.ndarray of shape (num_vecs, 3)
        or list of num_vecs lists of length 3).
        Can also be an arbitrary array with the last index having length 3.
    cell : array_like
        The lattice cell vectors.
        Can be given as a numpy.ndarray of shape (3,3) or as a list of three lists of length 3.
        By default (transpose=True), the array's rows are assumed to be the cell vectors,
        e.g. with cell[0] being the first cell vector.
        On the other hand, if transpose is set to False, the columns are assumed to be the cell
        vectors.
    transpose : bool, optional
        If True (default), cell is assumed to be provided with vectors as its rows,
        otherwise the cell vectors are assumed to be the cell array's columns.
    npa : bool, optional
        Only relevant if vec is 2D:
        If True (default), the output will be converted to a 2D numpy.ndarray. If False, the
        output will be a list of 1D numpy.ndarrays.

    Returns
    -------
    numpy.ndarray or list
        The Cartesian coordinates of the input vectors.
        If vec is 1D, then a numpy.ndarray of shape (3,) is returned.
        If vec is 2D and npa=True, a 2D numpy.ndarray of shape (num_vecs, 3) is returned.
        If vec is 2D and npa=False, a list of num_vecs numpy.ndarrays of shape (3,) is returned.

    """
    # Convert lists to numpy arrays;
    vec_npa = np.asarray(vec)
    cell_npa = np.array(cell)

    # If needed, transpose the cell array, such that columns are cell vectors
    if transpose:
        cell_npa = cell_npa.T

    # If only one vector is given, convert it
    if len(vec_npa.shape) == 1:
        out = cell_npa @ vec_npa
    # If multiple vectors are given, convert them one by one
    # If needed, this could easily be vectorized
    elif len(vec_npa.shape) == 2:
        N = vec_npa.shape[0]
        out = [cell_npa @ vec_npa[i, :] for i in range(N)]
    else:
        out = vec_npa @ cell_npa.T

    # Convert the output to a numpy array, if wanted
    if npa:
        out = np.asarray(out)
    return out


def cart_to_direct(vec, cell, transpose=True, npa=True):
    """
    Convert vectors from Cartesian to direct coordinates.

    This function converts a set of vectors from  Cartesian coordinates to direct (fractional)
    coordinates, using the provided (reciprocal) lattice cell vectors. It supports
    conversion of multiple vectors at once and allows for the lattice cell vectors to be input
    either as row vectors (default) or as column vectors.
    Ad Conversion: coords_cart = coords_frac_1 * lattice_vec_1 + ... (same for direction 2 and 3)
    This is the same as a matrix-vector multiplication, with the matrix's columns being
    the lattice vectors and the (column) vector containing the fractional coordinates.
    Thus, to go from cartesian to direct coordinates, we simply have to multiple the column vectors
    by the inverse of this matrix.

    Note
    ----
    `vec` and `cell` are assumed to have the same units.

    Parameters
    ----------
    vec : array_like
        The vectors to be converted, in cartesian coordinates. This can be a 1D array
        for a single vector (numpy.ndarray of shape (3,) or a list of length 3)
        or a 2D array, where each row represents a different vector and the last dimension
        corresponds to the spatial directions (numpy.ndarray of shape (num_vecs, 3)
        or list of num_vecs lists of length 3).
    cell : array_like
        The lattice cell vectors.
        Can be given as a numpy.ndarray of shape (3,3) or as a list of three lists of length 3.
        By default (transpose=True), the array's rows are assumed to be the cell vectors,
        e.g. with cell[0] being the first cell vector.
        On the other hand, if transpose is set to False, the columns are assumed to be the cell
        vectors.
    transpose : bool, optional
        If True (default), cell is assumed to be provided with vectors as its rows,
        otherwise the cell vectors are assumed to be the cell array's columns.
    npa : bool, optional
        Only relevant if vec is 2D:
        If True (default), the output will be converted to a 2D numpy.ndarray. If False, the
        output will be a list of 1D numpy.ndarrays.

    Returns
    -------
    numpy.ndarray or list
        The direct coordinates of the input vectors.
        If vec is 1D, then a numpy.ndarray of shape (3,) is returned.
        If vec is 2D and npa=True, a 2D numpy.ndarray of shape (num_vecs, 3) is returned.
        If vec is 2D and npa=False, a list of num_vecs numpy.ndarrays of shape (3,) is returned.

    """
    # Convert lists to numpy arrays;
    vec_npa = np.asarray(vec)
    cell_npa = np.array(cell)

    # If needed, transpose the cell array, such that columns are cell vectors
    if transpose:
        cell_npa = cell_npa.T

    # Invert the cell vector matrix
    cell_npa_inv = np.linalg.inv(cell_npa)

    # If only one vector is given, convert it
    if len(vec_npa.shape) == 1:
        out = cell_npa_inv @ vec_npa
    # If multiple vectors are given, convert them one by one
    # If needed, this could easily be vectorized
    elif len(vec_npa.shape) == 2:
        N = vec_npa.shape[0]
        out = [cell_npa_inv @ vec_npa[i, :] for i in range(N)]

    # Convert the output to a numpy array, if wanted
    if npa:
        out = np.asarray(out)
    return out


def generate_grid(cell, n1, n2, n3, cell_bounds=[0,1,0,1,0,1]):
    """
    Generate a 3D mesh of points within a parallelepiped defined by unit vectors.

    Parameters
    ----------
    vectors : ndarray of shape (3, 3)
        A 3x3 array where each row is a unit vector of the cell.
    n1, n2, n3 : int
        Number of voxels along each vector direction.
    cell_bounds : list of six int, optional, default: [0,1,0,1,0,1]
        Unit cell boundaries in each +/- direction.
        E.g., the first two numbers determine the boundaries in the 0th direction, i.e.
        -1,1 means, that the grid spans from -1*cell[0] to +1*cell[0].
        By default, only one unit cell is spanned.

    Returns
    -------
    mesh : ndarray of shape (n1, n2, n3, 3)
        Array containing the coordinates of each voxel.
    """
    # Create voxel indices
    x = np.linspace(cell_bounds[0], cell_bounds[1], n1, endpoint=False)
    y = np.linspace(cell_bounds[2], cell_bounds[3], n2, endpoint=False)
    z = np.linspace(cell_bounds[4], cell_bounds[5], n3, endpoint=False)

    # Create a 3D grid of indices
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing="ij")

    # Combine and transform using the unit vectors
    mesh = (
        grid_x[..., np.newaxis] * cell[0]
        + grid_y[..., np.newaxis] * cell[1]
        + grid_z[..., np.newaxis] * cell[2]
    )
    return mesh


def write_gaussian_cube(filename, atoms, data, line1="", line2=""):
    """
    Write a Gaussian cube file from an ASE atoms object and a 3D data array.

    The data is assumed to be calculated on a grid created by `generate_grid`.

    Parameters
    ----------
    filename : str
        Output file name.
    atoms : ase.Atoms
        ASE atoms object representing the atomic structure.
    data : np.ndarray
        3D numpy array of shape (nx, ny, nz) containing volumetric data.
    line1, line2 : str, str, optional
        First two (comment) lines to be written to the cube file.
    """
    # Get cell vectors and number of voxels along each vector
    cell = atoms.cell.array
    nx, ny, nz = data.shape
    # Get number of atoms
    natoms = len(atoms)

    with open(filename, "w") as f:
        # Header lines
        f.write(line1+"\n")
        f.write(line2+"\n")

        # Origin and number of atoms
        f.write(f"{natoms:5d} {0.0:12.6f} {0.0:12.6f} {0.0:12.6f}\n")
        # Unit cell vectors and grid points
        f.write(f"{-nx:5d} {cell[0, 0]:12.6f} {cell[0, 1]:12.6f} {cell[0, 2]:12.6f}\n")
        f.write(f"{-ny:5d} {cell[1, 0]:12.6f} {cell[1, 1]:12.6f} {cell[1, 2]:12.6f}\n")
        f.write(f"{-nz:5d} {cell[2, 0]:12.6f} {cell[2, 1]:12.6f} {cell[2, 2]:12.6f}\n")

        # Atomic coordinates
        for atom in atoms:
            charge = atom.number
            x, y, z = atom.position / Bohr
            f.write(f"{charge:5d} {0.0:12.6f} {x:12.6f} {y:12.6f} {z:12.6f}\n")

        # Data values
        data_flat = data.flatten()
        for i in range(0, len(data_flat), 6):
            line = " ".join(f"{val:13.5e}" for val in data_flat[i : i + 6])
            f.write(line + "\n")
