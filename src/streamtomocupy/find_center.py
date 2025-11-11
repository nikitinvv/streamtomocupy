import cupy as cp
import numpy as np
import cupyx.scipy.ndimage as ndimage


def find_center_vo(
    tomo, dark, flat, smin=-50, smax=50, srad=6, step=0.25, ratio=0.5, drop=20
):
    """
    Find rotation axis location using Nghia Vo's method. :cite:`Vo:14`.

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data or a 2D sinogram.
    ind : int, optional
        Index of the slice to be used for reconstruction.
    smin, smax : int, optional
        Coarse search radius. Reference to the horizontal center of
        the sinogram.
    srad : float, optional
        Fine search radius.
    step : float, optional
        Step of fine searching.
    ratio : float, optional
        The ratio between the FOV of the camera and the size of object.
        It's used to generate the mask.
    drop : int, optional
        Drop lines around vertical center of the mask.

    Returns
    -------
    float
        Rotation axis location.
    """

    _tomo = cp.array(tomo)
    _dark = cp.mean(cp.array(dark), axis=0)
    _flat = cp.mean(cp.array(flat), axis=0)
    _tomo = (_tomo - _dark) / (_flat - _dark + 1e-3)
    _tomo[_tomo <= 0] = 1
    _tomo[:] = -cp.log(_tomo)
    _tomo[cp.isnan(_tomo)] = 6.0
    _tomo[cp.isinf(_tomo)] = 0

    # Denoising
    # There's a critical reason to use different window sizes
    # between coarse and fine search.
    _tomo_cs = ndimage.gaussian_filter(_tomo, (3, 1), mode="reflect")
    _tomo_fs = ndimage.gaussian_filter(_tomo, (2, 2), mode="reflect")

    # Coarse and fine searches for finding the rotation center.
    if _tomo.shape[0] * _tomo.shape[1] > 4e4:  # If data is large (>2kx2k)
        _tomo_coarse = _downsample(_tomo_cs, level=2)
        init_cen = _search_coarse(_tomo_coarse, smin / 4.0, smax / 4.0, ratio, drop)
        fine_cen = _search_fine(_tomo_fs, srad, step, init_cen * 4.0, ratio, drop)
    else:
        init_cen = _search_coarse(_tomo_cs, smin, smax, ratio, drop)
        fine_cen = _search_fine(_tomo_fs, srad, step, init_cen, ratio, drop)
    return fine_cen


def _downsample(tomo, level):
    for k in range(level):
        tomo = 0.5 * (tomo[:-1:2] + tomo[1::2])
        tomo = 0.5 * (tomo[:, :-1:2] + tomo[:, 1::2])
    return tomo


def _calculate_metric(shift_col, sino1, sino2, sino3, mask):
    """
    Metric calculation.
    """
    shift_col = 1.0 * np.squeeze(shift_col)
    if np.abs(shift_col - np.floor(shift_col)) == 0.0:
        shift_col = int(shift_col)
        sino_shift = cp.roll(sino2, shift_col, axis=1)
        if shift_col >= 0:
            sino_shift[:, :shift_col] = sino3[:, :shift_col]
        else:
            sino_shift[:, shift_col:] = sino3[:, shift_col:]
        mat = cp.vstack((sino1, sino_shift))
    else:
        sino_shift = ndimage.shift(sino2, (0, shift_col), order=3, prefilter=True)
        if shift_col >= 0:
            shift_int = int(np.ceil(shift_col))
            sino_shift[:, :shift_int] = sino3[:, :shift_int]
        else:
            shift_int = int(np.floor(shift_col))
            sino_shift[:, shift_int:] = sino3[:, shift_int:]
        mat = cp.vstack((sino1, sino_shift))
    metric = cp.mean(cp.abs(cp.fft.fftshift(cp.fft.fft2(mat))) * mask)
    return metric


def _search_coarse(sino, smin, smax, ratio, drop):
    """
    Coarse search for finding the rotation center.
    """
    (nrow, ncol) = sino.shape
    cen_fliplr = (ncol - 1.0) / 2.0
    smin = np.int16(np.clip(smin + cen_fliplr, 0, ncol - 1) - cen_fliplr)
    smax = np.int16(np.clip(smax + cen_fliplr, 0, ncol - 1) - cen_fliplr)
    start_cor = ncol // 2 + smin
    stop_cor = ncol // 2 + smax
    flip_sino = cp.fliplr(sino)
    comp_sino = cp.flipud(sino)  # Used to avoid local minima
    list_cor = np.arange(start_cor, stop_cor + 0.5, 0.5)
    list_metric = np.zeros(len(list_cor), dtype=np.float32)
    mask = _create_mask(2 * nrow, ncol, 0.5 * ratio * ncol, drop)
    list_shift = 2.0 * (list_cor - cen_fliplr)

    for k, s in enumerate(list_shift):
        list_metric[k] = _calculate_metric(s, sino, flip_sino, comp_sino, mask)
    minpos = np.argmin(list_metric)
    if minpos == 0:
        print("WARNING!!!Global minimum is out of searching range")
        print("Please extend smin: %i", smin)
    if minpos == len(list_metric) - 1:
        print("WARNING!!!Global minimum is out of searching range")
        print("Please extend smax: %i", smax)
    cor = list_cor[minpos]
    return cor


def _search_fine(sino, srad, step, init_cen, ratio, drop):
    """
    Fine search for finding the rotation center.
    """
    (nrow, ncol) = sino.shape
    cen_fliplr = (ncol - 1.0) / 2.0
    srad = np.clip(np.abs(srad), 1.0, ncol / 4.0)
    step = np.clip(np.abs(step), 0.1, srad)
    init_cen = np.clip(init_cen, srad, ncol - srad - 1)

    list_cor = init_cen + np.arange(-srad, srad + step, step)

    flip_sino = cp.fliplr(sino)
    comp_sino = cp.flipud(sino)
    mask = _create_mask(2 * nrow, ncol, 0.5 * ratio * ncol, drop)
    list_shift = 2.0 * (list_cor - cen_fliplr)
    list_metric = np.zeros(len(list_cor), dtype=np.float32)
    for k, s in enumerate(list_shift):
        list_metric[k] = _calculate_metric(s, sino, flip_sino, comp_sino, mask)
    cor = list_cor[np.argmin(list_metric)]
    return cor


def _create_mask(nrow, ncol, radius, drop):
    """
    Make a binary mask to select coefficients outside the double-wedge region.
    Eq.(3) in https://doi.org/10.1364/OE.22.019078

    Parameters
    ----------
    nrow : int
        Image height.
    ncol : int
        Image width.
    radius: int
        Radius of an object, in pixel unit.
    drop : int
        Drop lines around vertical center of the mask.

    Returns
    -------
        2D binary mask.
    """
    du = 1.0 / ncol
    dv = (nrow - 1.0) / (nrow * 2.0 * np.pi)
    cen_row = np.int16(np.ceil(nrow / 2.0) - 1)
    cen_col = np.int16(np.ceil(ncol / 2.0) - 1)
    drop = min(drop, np.int16(np.ceil(0.05 * nrow)))
    mask = cp.zeros((nrow, ncol), dtype="float32")
    for i in range(nrow):
        pos = np.int16(np.ceil(((i - cen_row) * dv / radius) / du))
        (pos1, pos2) = np.clip(np.sort((-pos + cen_col, pos + cen_col)), 0, ncol - 1)
        mask[i, pos1 : pos2 + 1] = 1.0
    mask[cen_row - drop : cen_row + drop + 1, :] = 0.0
    mask[:, cen_col - 1 : cen_col + 2] = 0.0
    return mask
