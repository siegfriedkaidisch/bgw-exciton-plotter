import numpy as np
import h5py

"""
Read data from a Berkely-GW calculation.
"""


def read_bse_eigenvectors(
    fname, n_max=20, meta_only=False, ev_only=True, analyze=False
):
    """
    Reads eigenvectors from a BerkeleyGW Absorption calculation and returns them
    alongside some additional data.

    Supported Calculations/Files
    ----------------------------
    - Spin-Unpolarized: one energy per band and q (num_channels=1, 2 electrons per scalar wavefunction)
    - Spin-Polarized, collinear: two energies per band and q (num_channels=2, 1 electron per up/down scalar wavefunction)
    - Noncollinear: one energy per band and q (num_channels=1, 1 electron per 2-spinor wavefunction)

    Parameters
    ----------
    fname : str
        Name of file containing the data, usually `eigenvectors.h5`.
    n_max : int, optional
        How many eigenvectors to return, starting from low energies; default: 20
        If you want to receive all eigenvectors, set this to `None`.
        Assumes, that eigenvectors are sorted by energy.
        Only relevant, if `ev_only` is True.
    meta_only : bool, optional
        True -> Only return some metadata about calculation, no eigenvectors; default: False
    ev_only : bool, optional
        Only relevant, if `meta_only` is False.
        True -> return eigenvectors and metadata; default: True
        False -> return eigenvectors, metadata and some additional data
    analyze : bool, optional
        To be used for debugging/inspecting data, default: False

    Returns :
    ---------
    Depending on the input flags, a selection of the following:

    meta : dict
        Contents:
        - "kpoints" : dict
            - exciton_Q_shifts : numpy.ndarray of shape (nQ, 3); MINUS exciton c.o.m momenta in ???? coordinates/units
            - kpts : numpy.ndarray of shape (nk, 3); k-points in direct coordinates
            - nQ : int; number of Q-points
            - nk : int; number of k-points
        - "params" : dict; see [1]

    ev : numpy.ndarray of shape (nQ, n_max, num_channels, nv, nc, nk)
        The normalized BSE eigenvectors A^(Q,m)_(s,v,c,k).
        Indices:
        - Q
        - excitation number, m
        - spin channel (if two, spc, then index 0 is spin-up and 1 is spin-down)
        - valence band  (counted from gap)
        - conduction band (counted from gap)
        - k-point

    data : dict; see [1]

    Return logic :
        - meta_only=True -> meta
        - meta_only=False and ev_only=True -> ev, meta (default)
        - meta_only=False and ev_only=False -> data, meta

    References
    ----------
    [1] http://manual.berkeleygw.org/4.0/eigenvectors_h5_spec/

    """

    with h5py.File(fname, "r") as f:
        # Store metadata in a dictionary
        meta = dict()

        # Read q-point data
        x = "kpoints"
        meta[x] = dict()
        for key in f["exciton_header"][x]:
            meta[x][key] = f["exciton_header"][x][key][()]

        # Read information about the BSE calculation
        x = "params"
        meta[x] = dict()
        meta[x]["flavor"] = f["exciton_header"]["flavor"][()]
        meta[x]["versionnumber"] = f["exciton_header"]["versionnumber"][()]
        for key in f["exciton_header"][x]:
            meta[x][key] = f["exciton_header"][x][key][()]

        # If wanted, return metadata only
        if meta_only:
            return meta

        # If wanted, return eigenvectors and metadata only
        if ev_only:
            # Reshape eigenvectors
            the_shape = (
                meta["kpoints"]["nQ"],
                meta["params"]["nevecs"],
                meta["kpoints"]["nk"],
                meta["params"]["nc"],
                meta["params"]["nv"],
                meta["params"]["ns"],
                meta["params"]["flavor"],
            )
            ev = np.reshape(f["exciton_data"]["eigenvectors"][()], the_shape)
            # norm = np.sum(np.abs(ev[:,,::])**2)

            # only return low-lying eigenvectors
            ev = ev[:, :n_max, ::]

            # convert to complex array, if flavor=2
            if meta["params"]["flavor"] == 2:
                ev = ev[:, :, :, :, :, :, 0] + 1j * ev[:, :, :, :, :, :, 1]
            elif meta["params"]["flavor"] == 1:
                ev = ev[:, :, :, :, :, :, 0]
            else:
                raise Exception("Unknown flavor: ", meta["params"]["flavor"])

            # swap some indices/axes
            ev = np.moveaxis(ev, [0, 1, 2, 3, 4, 5], [0, 1, 5, 4, 3, 2])

            # See what is going on
            if analyze:
                n_Q = 0
                n_vec = 3
                n_s = 0
                n_v = 0  # count from gap
                n_c = 0  # count from gap
                test = np.reshape(ev[n_Q, n_vec, n_s, n_v, n_c, :], (48, 48))
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots()
                im = ax.imshow(np.real(np.abs(test**2)), origin="lower")
                plt.colorbar(im)

                plt.show()
                exit()

            return ev, meta

        # Otherwise, read and return additional data
        else:
            data = dict()
            for key in list(f["exciton_data"]):
                data[key] = f["exciton_data"][key][()]

            return data, meta
