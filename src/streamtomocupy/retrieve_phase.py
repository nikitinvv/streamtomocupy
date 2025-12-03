"""Paganin phase retrieval implementation"""

import cupy as cp
from cupy.fft import fft2, ifft2

__all__ = [
    "paganin_filter",
]

BOLTZMANN_CONSTANT = 1.3806488e-16  # [erg/k]
SPEED_OF_LIGHT = 299792458e2  # [cm/s]
PI = 3.14159265359
PLANCK_CONSTANT = 6.58211928e-19  # [keV*s]


def _wavelength(energy):
    return 2 * PI * PLANCK_CONSTANT * SPEED_OF_LIGHT / energy


def paganin_filter(
    data,
    pixel_size=1e-4,
    dist=50,
    energy=20,
    alpha=1e-3,
    method="paganin",
    db=1000,
    W=2e-4,
    pad=True,
):
    """
    Perform single-step phase retrieval from phase-contrast measurements
    :cite:`Paganin:02`.

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    pixel_size : float, optional
        Detector pixel size in cm.
    dist : float, optional
        Propagation distance of the wavefront in cm.
    energy : float, optional
        Energy of incident wave in keV.
    alpha : float, optional
        Regularization parameter for Paganin method.
    method : string
        phase retrieval method. Standard Paganin or Generalized Paganin.
    db : float, optional
        delta/beta for generalized Paganin phase retrieval
    W :  float
        Characteristic transverse lenght scale
    pad : bool, optional
        If True, extend the size of the projections by padding with zeros.
    Returns
    -------
    ndarray
        Approximated 3D tomographic phase data.
    """
    # New dimensions and pad value after padding.
    py, pz = _calc_pad(data, pixel_size, dist, energy, pad)

    # Compute the reciprocal grid.
    dx, dy, dz = data.shape
    if method == "paganin":
        w2 = _reciprocal_grid(pixel_size, dy + 2 * py, dz + 2 * pz)
        phase_filter = cp.fft.fftshift(_paganin_filter_factor(energy, dist, alpha, w2))
    elif method == "Gpaganin":
        kf = _reciprocal_gridG(pixel_size, dy + 2 * py, dz + 2 * pz)
        phase_filter = cp.fft.fftshift(
            _paganin_filter_factorG(energy, dist, kf, pixel_size, db, W)
        )

    prj = cp.empty([dy + 2 * py, dz + 2 * pz], dtype=data.dtype)
    
    _retrieve_phase(data, phase_filter, py, pz, prj, pad)

    return data


def _retrieve_phase(data, phase_filter, px, py, prj, pad):
    dx, dy, dz = data.shape
    num_jobs = data.shape[0]
    normalized_phase_filter = phase_filter / phase_filter.max()    
    
    for m in range(num_jobs):        
        prj[:] = cp.pad(data[m],((px,px),(py,py)),mode='edge')
        fproj = fft2(prj)
        fproj *= normalized_phase_filter
        proj = cp.real(ifft2(fproj))
        if pad:
            proj = proj[px : dy + px, py : dz + py]
        data[m] = proj


def _calc_pad(data, pixel_size, dist, energy, pad):
    """
    Calculate new dimensions and pad value after padding.

    Parameters
    ----------
    data : ndarray
        3D tomographic data.
    pixel_size : float
        Detector pixel size in cm.
    dist : float
        Propagation distance of the wavefront in cm.
    energy : float
        Energy of incident wave in keV.
    pad : bool
        If True, extend the size of the projections by padding with zeros.

    Returns
    -------
    int
        Pad amount in projection axis.
    int
        Pad amount in sinogram axis.
    float
        Pad value.
    """
    dx, dy, dz = data.shape
    wavelength = _wavelength(energy)
    py, pz, = 0, 0
    if pad:
        py = _calc_pad_width(dy, pixel_size, wavelength, dist)
        pz = _calc_pad_width(dz, pixel_size, wavelength, dist)

    return py, pz


def _paganin_filter_factor(energy, dist, alpha, w2):
    return 1 / (_wavelength(energy) * dist * w2 / (4 * PI) + alpha)


def _paganin_filter_factorG(energy, dist, kf, pixel_size, db, W):
    """
    Generalized phase retrieval method
    Paganin et al 2020
    diffracting feature ~2*pixel size
    """
    aph = db * (dist * _wavelength(energy)) / (4 * PI)
    return 1 / (1.0 - (2 * aph / (W**2)) * (kf - 2))


def _calc_pad_width(dim, pixel_size, wavelength, dist):
    pad_pix = int(cp.ceil(PI * wavelength * dist / pixel_size ** 2))
    return (2**int(cp.ceil(cp.log2(dim + pad_pix))) - dim) // 2


def _reciprocal_grid(pixel_size, nx, ny):
    """
    Calculate reciprocal grid.

    Parameters
    ----------
    pixel_size : float
        Detector pixel size in cm.
    nx, ny : int
        Size of the reciprocal grid along x and y axes.

    Returns
    -------
    ndarray
        Grid coordinates.
    """
    # Sampling in reciprocal space.
    indx = _reciprocal_coord(pixel_size, nx)
    indy = _reciprocal_coord(pixel_size, ny)
    cp.square(indx, out=indx)
    cp.square(indy, out=indy)

    idx, idy = cp.meshgrid(indy, indx)
    return idx + idy


def _reciprocal_gridG(pixel_size, nx, ny):
    """
    Calculate reciprocal grid for Generalized Paganin method.

    Parameters
    ----------
    pixel_size : float
        Detector pixel size in cm.
    nx, ny : int
        Size of the reciprocal grid along x and y axes.

    Returns
    -------
    ndarray
        Grid coordinates.
    """
    # Considering diffracting feature ~2*pixel size
    # Sampling in reciprocal space.
    indx = cp.cos(_reciprocal_coord(pixel_size, nx) * 2 * PI * pixel_size)
    indy = cp.cos(_reciprocal_coord(pixel_size, ny) * 2 * PI * pixel_size)
    idx, idy = cp.meshgrid(indy, indx)
    return idx + idy


def _reciprocal_coord(pixel_size, num_grid):
    """
    Calculate reciprocal grid coordinates for a given pixel size
    and discretization.

    Parameters
    ----------
    pixel_size : float
        Detector pixel size in cm.
    num_grid : int
        Size of the reciprocal grid.

    Returns
    -------
    ndarray
        Grid coordinates.
    """
    rc = cp.fft.fftshift(cp.fft.fftfreq(num_grid,d=pixel_size))
    return rc
