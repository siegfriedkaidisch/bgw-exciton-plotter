import numpy as np
import h5py
import os
import re
from copy import deepcopy
import xml.etree.ElementTree as ET
import warnings

from library.misc import cart_to_direct

"""
Read energies and wavefunctions from QE calculations
"""


def read_wavefunction_q_qe(wfn_folder, iqpt, ibands="all", meta_only=False):
    """
    Extract Miller indices and wavefunction coefficients (+ some metadata) for a given
    q-point-index and list of bands from a Quantum Espresso calculation.

    Reads Quantum Espresso wave function data from an HDF5 or dat file and returns a
    dictionary with metadata and optionally, if meta_only=False, an additional
    dictionary with G-vectors (Miller indices) and basis coefficients.

    Supported Calculations/Files
    ----------------------------
    - Spin-Unpolarized: 2 electrons per scalar wavefunction (up/down wavefunctions are identical)
    - Spin-Polarized, collinear: 1 electron per up/down scalar wavefunction
    - Noncollinear: 1 electron per 2-spinor wavefunction

    Parameters
    ----------
    wfn_folder : str
        Path to folder containing wavefunction files (dat or hdf5).
    iqpt : int
        Index of the q-point (1-based index).
    ibands : "all" or list of int, optional
        By default, all bands are loaded ("all").
        You can also just load certain bands to save memory (0-based).
        (If a list is given, it must not contain the same integer twice!
        I.e., don't load the same band twice in one go!)
    meta_only : bool, optional
        If True, the function returns only metadata. Default is False.

    Returns
    -------
    dict or tuple
        If `meta_only` is False, returns a tuple containing two dictionaries: (wf, meta).
        - `wf` includes:
            - `miller`: numpy.ndarray, shape (nG, 3), Miller indices (integers).
            - `cnq`: numpy.ndarray, shape (nspin, # of loaded bands, nG), complex wave function coefficients.
                - "su": `nspin`=1
                - "spc": `nspin`=2, cnq[0]= "up" scalar wavefunction, cnq[1]= "down" scalar wavefunction
                - "spn": `nspin`=2, cnq[0]=up and cnq[1]=down component of spinor wavefunction
        - `meta` includes:
            - `nbands`: int, number of bands present in the hdf5/dat file.
            - `ibands`: "all" or list of int, indices of loaded bands (only loaded if meta_only=False)
            - `spin_mode`: "su", "spc" or "spn"
            - `nG`: int, number of G-vectors.
            - `q`: numpy.ndarray, shape (3,), q-vector of this file in direct coordinates.
            - `b`: numpy.ndarray, shape (3, 3), reciprocal lattice vectors (rows are vectors, units: 1/Bohr).
        If `meta_only` is True, returns only the meta dictionary.

    """
    # Catch bad inputs
    if ibands != "all":
        if len(ibands) != len(set(ibands)):
            raise Exception("`ibands` contains duplicate elements!")
    # Read wavefunction differently, depending on file format and spin-mode
    if os.path.isfile(wfn_folder + "wfc" + str(iqpt) + ".dat"):
        # spc = False
        # format = "dat"
        return read_wavefunction_q_qe_dat(
            dat_file=wfn_folder + "wfc" + str(iqpt) + ".dat",
            meta_only=meta_only,
            ibands=ibands,
        )
    elif os.path.isfile(wfn_folder + "wfc" + str(iqpt) + ".hdf5"):
        # spc = False
        # format = "hdf5"
        return read_wavefunction_q_qe_hdf5(
            hdf5_file=wfn_folder + "wfc" + str(iqpt) + ".hdf5",
            meta_only=meta_only,
            ibands=ibands,
        )
    elif os.path.isfile(wfn_folder + "wfcup" + str(iqpt) + ".dat"):
        # spc = True
        # format = "dat"
        tmp_up = read_wavefunction_q_qe_dat(
            dat_file=wfn_folder + "wfcup" + str(iqpt) + ".dat",
            meta_only=meta_only,
            ibands=ibands,
        )
        if meta_only:
            return tmp_up
        else:
            tmp_dn = read_wavefunction_q_qe_dat(
                dat_file=wfn_folder + "wfcdn" + str(iqpt) + ".dat",
                meta_only=meta_only,
                ibands=ibands,
            )
            meta = tmp_up[1]
            wf = dict()
            wf["miller"] = tmp_up[0]["miller"]
            wf["cnq"] = np.concatenate([tmp_up[0]["cnq"], tmp_dn[0]["cnq"]], axis=0)
            return wf, meta
    elif os.path.isfile(wfn_folder + "wfcup" + str(iqpt) + ".hdf5"):
        # spc = True
        # format = "hdf5"
        tmp_up = read_wavefunction_q_qe_hdf5(
            hdf5_file=wfn_folder + "wfcup" + str(iqpt) + ".hdf5",
            meta_only=meta_only,
            ibands=ibands,
        )
        if meta_only:
            return tmp_up
        else:
            tmp_dn = read_wavefunction_q_qe_hdf5(
                hdf5_file=wfn_folder + "wfcdn" + str(iqpt) + ".hdf5",
                meta_only=meta_only,
                ibands=ibands,
            )
            meta = tmp_up[1]
            wf = dict()
            wf["miller"] = tmp_up[0]["miller"]
            wf["cnq"] = np.concatenate([tmp_up[0]["cnq"], tmp_dn[0]["cnq"]], axis=0)
            return wf, meta
    else:
        raise ValueError(
            "Could not detect type of WFN files."
            + " The names of the files must contain `wfc` or `wfcup/wfcdn` and end in '.hdf5' or '.dat'!"
        )


def read_wavefunction_q_qe_hdf5(hdf5_file, ibands="all", meta_only=False):
    """
    Read wavefunction from a QE HDF5 file.

    For details see `read_wavefunction_q_qe`.

    References
    ----------
    - https://lists.quantum-espresso.org/pipermail/users/2008-October/010259.html
    - https://gitlab.com/QEF/q-e/-/snippets/1869219

    """

    with h5py.File(hdf5_file, "r") as f:
        # for key in list(f.attrs):
        #    print(key, f.attrs[key])
        meta = dict()

        # Save number of available bands and the requested bands as metadata
        meta["nbands"] = f.attrs["nbnd"]
        meta["ibands"] = deepcopy(ibands)

        # Get number of G vectors, int
        meta["nG"] = f.attrs["igwx"]

        # Get q vector of this file, numpy.ndarray of shape (3,)
        # Cartesian Coordinates, in 1/Bohr
        meta["q"] = np.array(f.attrs["xk"])

        # Check if calculation is spc
        spc = True if "wfcup" in hdf5_file or "wfcdn" in hdf5_file else False
        # Check if calculation is non-collinear
        spn = True if f.attrs["npol"] == 2 else False
        # Get spin_mode from that
        if spc:
            spin_mode = "spc"
        elif spn:
            spin_mode = "spn"
        else:
            spin_mode = "su"
        meta["spin_mode"] = spin_mode

        # Get reciprocal cell. numpy.ndarray of shape (3,3)
        # rows as cell vectors (i.e. meta["b"][0] is first vector)
        # Units: 1/Bohr
        b = [
            list(f["MillerIndices"].attrs["bg1"]),
            list(f["MillerIndices"].attrs["bg2"]),
            list(f["MillerIndices"].attrs["bg3"]),
        ]
        meta["b"] = np.asarray(b)

        # Convert q to direct coordinates
        meta["q"] = cart_to_direct(vec=meta["q"], cell=meta["b"])

        if meta_only:
            return meta
        else:
            wf = dict()

            # Get Miller indices, numpy.ndarray of shape (ng, 3)
            # contains integers
            wf["miller"] = np.array(f["MillerIndices"])

            # get coeffs. numpy.ndarray of shape (npol,nb,ng)
            # How is it normalized??
            if ibands == "all":
                evc = np.asarray(f["evc"])
                nbands2load = meta["nbands"]
            else:
                evc = np.asarray(f["evc"][ibands])
                nbands2load = len(ibands)
            if spin_mode == "spn":
                evc = evc.reshape((nbands2load, meta["nG"] * 2, 2))
                wf["cnq"] = np.array([evc[:, : meta["nG"], :], evc[:, meta["nG"] :, :]])
            else:
                wf["cnq"] = evc.reshape((1, nbands2load, meta["nG"], 2))
            # convert from two real to one complex array
            wf["cnq"] = wf["cnq"][:, :, :, 0] + 1j * wf["cnq"][:, :, :, 1]
            return wf, meta


def read_wavefunction_q_qe_dat(dat_file, ibands="all", meta_only=False):
    """
    Read wavefunction from a QE dat file.

    For details see `read_wavefunction_q_qe`.

    References
    ----------
    - https://lists.quantum-espresso.org/pipermail/users/2008-October/010259.html
    - https://mattermodeling.stackexchange.com/questions/9149/how-to-read-qes-wfc-dat-files-with-python

    """

    with open(dat_file, "rb") as f:
        # Moves the cursor 4 bytes to the right
        f.seek(4)

        _ = np.fromfile(f, dtype="int32", count=1)[0]  # ik
        q = np.fromfile(f, dtype="float64", count=3)
        _ = np.fromfile(f, dtype="int32", count=1)[0]  # ispin = 0 or 1
        _ = bool(np.fromfile(f, dtype="int32", count=1)[0])  # gamma_only
        _ = np.fromfile(f, dtype="float64", count=1)[0]  # scalef

        f.seek(8, 1)

        _ = np.fromfile(f, dtype="int32", count=1)[0]  # ngw
        nG = np.fromfile(f, dtype="int32", count=1)[0]
        npol = np.fromfile(f, dtype="int32", count=1)[0]
        nbands = np.fromfile(f, dtype="int32", count=1)[0]

        f.seek(8, 1)

        b1 = np.fromfile(f, dtype="float64", count=3)
        b2 = np.fromfile(f, dtype="float64", count=3)
        b3 = np.fromfile(f, dtype="float64", count=3)
        b = np.array([b1, b2, b3])

        # Collect metadata
        meta = dict()
        meta["nbands"] = nbands
        meta["ibands"] = deepcopy(ibands)
        meta["nG"] = nG
        meta["q"] = cart_to_direct(vec=q, cell=b)
        meta["b"] = b

        # Check if calculation is spc
        spc = True if "wfcup" in dat_file or "wfcdn" in dat_file else False
        # Check if calculation is non-collinear
        spn = True if npol == 2 else False
        # Get spin_mode from that
        if spc:
            spin_mode = "spc"
        elif spn:
            spin_mode = "spn"
        else:
            spin_mode = "su"
        meta["spin_mode"] = spin_mode

        if meta_only:
            return meta
        else:
            wf = dict()
            f.seek(8, 1)

            # Get Miller indices, numpy.ndarray of shape (ng, 3)
            # contains integers
            miller = np.fromfile(f, dtype="int32", count=3 * nG)
            miller = miller.reshape((nG, 3))
            wf["miller"] = miller

            # Get wavefunctions of shape (npol,nb,ng)
            if ibands == "all":
                bands2load = list(range(nbands))
            else:
                bands2load = ibands
                if np.max(bands2load) > nbands - 1:
                    raise Exception("Not all requested bands are available in file!")
            nbands2load = len(bands2load)
            cnq = np.zeros((nbands2load, npol * nG), dtype="complex128")
            f.seek(8, 1)
            for i in range(nbands):
                if i in bands2load:
                    j = np.where(np.array(bands2load) == i)[0][0]
                    cnq[j, :] = np.fromfile(f, dtype="complex128", count=npol * nG)
                    f.seek(8, 1)
                else:
                    f.seek(npol * nG * 16 + 8, 1)
            # (nbands2load, npol*nG) -> (npol, nbands2load, nG)
            if npol == 2:
                cnq = np.array([cnq[:, :nG], cnq[:, nG:]])
            elif npol == 1:
                cnq = cnq.reshape((1, nbands2load, nG))
            wf["cnq"] = cnq
            return wf, meta


def find_min_max_iq_qe(wfn_folder):
    """
    Find number of available q-point wavefunctions.

    Find the minimum and maximum values of i_q from filenames in a specified folder
    containing files with this naming convention:
    - su or spn: wf_name = wfn_folder + "/wfc" + str(i_q) + ".hdf5/dat"
    - spc: wf_name = wfn_folder + "/wfcup" + str(i_q) + ".hdf5/dat"

    Supported Calculations/Files
    ----------------------------
    - Spin-Unpolarized
    - Spin-Polarized, collinear
    - Noncollinear

    (The function first searches for the su/spn wfc files. If it cannot find any, it searches
    for the spc wfcup files.)

    Parameters
    ----------
    wfn_folder : str
        The path to the folder containing the files.

    Returns
    -------
    tuple
        A tuple containing the minimum and maximum values of i_q.
    """
    # Patterns to extract the number i_q from the filename
    pattern1a = re.compile(r"wfc(\d+)\.hdf5$")
    pattern1b = re.compile(r"wfc(\d+)\.dat$")
    pattern2a = re.compile(r"wfcup(\d+)\.hdf5$")
    pattern2b = re.compile(r"wfcup(\d+)\.dat$")

    # Lists to store extracted numbers
    i_q_values_pattern1 = []
    i_q_values_pattern2 = []

    # Loop over all files in the directory
    for filename in os.listdir(wfn_folder):
        # Check if the filename matches the patterns
        match1a = pattern1a.search(filename)
        match1b = pattern1b.search(filename)
        match2a = pattern2a.search(filename)
        match2b = pattern2b.search(filename)
        if match1a:
            i_q_values_pattern1.append(int(match1a.group(1)))
        elif match1b:
            i_q_values_pattern1.append(int(match1b.group(1)))
        if match2a:
            i_q_values_pattern2.append(int(match2a.group(1)))
        elif match2b:
            i_q_values_pattern2.append(int(match2b.group(1)))

    if len(i_q_values_pattern1) > 0 and len(i_q_values_pattern2) > 0:
        raise Exception(
            "There are incompatible wavefunctions (su/spn and spc) in "
            + str(wfn_folder)
        )
    elif len(i_q_values_pattern1) > 0:
        return min(i_q_values_pattern1), max(i_q_values_pattern1)
    elif len(i_q_values_pattern2) > 0:
        return min(i_q_values_pattern2), max(i_q_values_pattern2)
    else:
        return None, None  # In case no matching files are found


def get_number_of_valence_bands_qe(fname):
    """
    Get number of valence bands calculated in a Quantum Espresso calculation.

    Supported Calculations/Assumptions
    ----------------------------------
    - System must be insulator or semiconductor (we need a finite bandgap) -> full valence
      bands, empty conduction bands
    - In case of a spin-polarized, collinear (spc) calculation, the number of
      valence bands in the spin up/down channels must be equal. -> even # of electrons
    - Except for these restrictions, all types of spin-treatment (su, spc, spn)
      are supported.

    Parameters
    ----------
    fn : str
        The file containing the energy data, xml file

    Returns
    -------
    int
        - The number of valence bands, which means:
        - Spin-Unpolarized: number of bands, with 2 electrons per band
        - Spin-polarized, collinear: number of bands per spin channel (up/down)
        - Noncollinear: number of bands, with 1 electron per band

    """
    msg1 = (
        " Please make sure, that your system is an insulator or a semiconductor, "
        + "such that all valence bands are completely filled "
        + "and all conduction bands are completely empty in the Quantum Espresso calculation."
    )

    if not ".xml" in fname.lower():
        warnings.warn(
            "Did you select the correct file for band energies? '.xml' not found in provided filename!",
            UserWarning,
        )
    # Load and parse the XML file
    tree = ET.parse(fname)
    root = tree.getroot()
    # Check, if calc was nspin=2 (LSDA/spc)
    spc = root.findall(".//lsda")[0].text == "true"
    # Check, if calc was nspin=4 (Noncollinear)
    noncolin = root.findall(".//noncolin")[0].text == "true"
    if spc and noncolin:
        raise Exception("Calculation was spc and Noncollinear at the same time???")

    band_structure = root.find("output/band_structure")
    for ks in band_structure.findall("ks_energies"):
        # Extract occupations (convert string to float list)
        occupations = ks.find("occupations").text.strip().split()
        occupations = np.round([float(val) for val in occupations], 9)
        sum_occupations = np.sum(occupations)
        # Sanity Checks
        if np.abs(np.round(sum_occupations) - sum_occupations) > 1e-6:
            warnings.warn(
                "Sum of occupations should be integer, but it is:",
                sum_occupations,
                UserWarning,
            )
        dev_from_int = np.max(np.abs(occupations - np.round(occupations)))
        if dev_from_int > 1e-6:
            warnings.warn(
                "Occupations should be integers, but they deviate from integer values by up to:",
                dev_from_int,
                UserWarning,
            )

        if spc:  # nspin=2
            # Check # of electrons
            if (np.round(sum_occupations) % 2) != 0:
                warnings.warn(
                    "spc calculation detected, but number of electrons is odd!",
                    UserWarning,
                )
            # Get # of bands
            num_valence_bands_dft = int(np.round(sum_occupations) / 2)
            num_conduction_bands_dft = len(occupations) / 2 - num_valence_bands_dft
            # Check occupations, expect (1,1,...1, 0,0...0, 1,1,...1, 0,0,...0)
            expected_occupations = (
                [1.0 for _ in num_valence_bands_dft]
                + [0.0 for _ in num_conduction_bands_dft]
                + [1.0 for _ in num_valence_bands_dft]
                + [0.0 for _ in num_conduction_bands_dft]
            )
            if not np.array_equal(occupations, expected_occupations):
                warnings.warn("spc occupations not as expected! " + msg1, UserWarning)
        elif noncolin:  # nspin=4
            # Get # of bands
            num_valence_bands_dft = int(np.round(sum_occupations))
            num_conduction_bands_dft = len(occupations) - num_valence_bands_dft
            # Check occupations, expect (1,1,...1, 0,0...0)
            expected_occupations = [1.0 for _ in num_valence_bands_dft] + [
                0.0 for _ in num_conduction_bands_dft
            ]
            if not np.array_equal(occupations, expected_occupations):
                warnings.warn(
                    "Noncollinear occupations not as expected! " + msg1, UserWarning
                )
        else:  # nspin=1
            # Get # of bands
            num_valence_bands_dft = int(np.round(sum_occupations))
            num_conduction_bands_dft = int(len(occupations) - num_valence_bands_dft)
            # Check occupations, expect (2,2,...2, 0,0...0)
            expected_occupations = [1.0 for _ in range(num_valence_bands_dft)] + [
                0.0 for _ in range(num_conduction_bands_dft)
            ]
            if not np.array_equal(occupations, expected_occupations):
                warnings.warn(
                    "Spin-unpolarized occupations not as expected! " + msg1,
                    UserWarning,
                )

    return int(num_valence_bands_dft)