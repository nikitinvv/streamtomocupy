import cupy as cp
import pywt
from cupyx.scipy.ndimage import median_filter
from cupyx.scipy.ndimage import binary_dilation
from cupyx.scipy.ndimage import uniform_filter1d
from streamtomocupy.utils import conv2d_kernel, conv_transpose2d_kernel

__all__ = [
    "DWTForward",
    "DWTInverse",
    "afb1d",
    "remove_all_stripe",
    "remove_stripe_fw",
    "remove_stripe_ti",
]

###### Ring removal with wavelet filtering (adapted for cupy from pytroch_wavelet package https://pytorch-wavelets.readthedocs.io/)##########


def _reflect(x, minx, maxx):
    """Reflect the values in matrix *x* about the scalar values *minx* and
    *maxx*.  Hence a vector *x* containing a long linearly increasing series is
    converted into a waveform which ramps linearly up and down between *minx*
    and *maxx*.  If *x* contains integers and *minx* and *maxx* are (integers +
    0.5), the ramps will have repeated max and min samples.

    .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
    .. codeauthor:: Nick Kingsbury, Cambridge University, January 1999.

    """
    x = cp.asanyarray(x)
    rng = maxx - minx
    rng_by_2 = 2 * rng
    mod = cp.fmod(x - minx, rng_by_2)
    normed_mod = cp.where(mod < 0, mod + rng_by_2, mod)
    out = cp.where(normed_mod >= rng, rng_by_2 - normed_mod, normed_mod) + minx
    return cp.array(out, dtype=x.dtype)


def _mypad(x, pad, value=0):
    """Function to do numpy like padding on Arrays. Only works for 2-D
    padding.

    Inputs:
        x (array): Array to pad
        pad (tuple): tuple of (left, right, top, bottom) pad sizes
    """
    # Vertical only
    if pad[0] == 0 and pad[1] == 0:
        m1, m2 = pad[2], pad[3]
        l = x.shape[-2]
        xe = _reflect(cp.arange(-m1, l + m2, dtype="int32"), -0.5, l - 0.5)
        return x[:, :, xe]
    # horizontal only
    elif pad[2] == 0 and pad[3] == 0:
        m1, m2 = pad[0], pad[1]
        l = x.shape[-1]
        xe = _reflect(cp.arange(-m1, l + m2, dtype="int32"), -0.5, l - 0.5)
        return x[:, :, :, xe]


def _conv2d(x, w, stride, pad, groups=1):
    """Convolution (equivalent pytorch.conv2d)"""
    if pad != 0:
        x = cp.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), "constant")

    b, ci, hi, wi = x.shape
    co, _, hk, wk = w.shape
    ho = int(cp.floor(1 + (hi - hk) / stride[0]))
    wo = int(cp.floor(1 + (wi - wk) / stride[1]))
    out = cp.zeros([b, co, ho, wo], dtype="float32")
    x = cp.expand_dims(x, axis=1)
    w = cp.expand_dims(w, axis=0)
    chunk = ci // groups
    chunko = co // groups
    for g in range(groups):
        for ii in range(hk):
            for jj in range(wk):
                x_windows = x[
                    :,
                    :,
                    g * chunk : (g + 1) * chunk,
                    ii : ho * stride[0] + ii : stride[0],
                    jj : wo * stride[1] + jj : stride[1],
                ]
                out[:, g * chunko : (g + 1) * chunko] += cp.sum(
                    x_windows
                    * w[:, g * chunko : (g + 1) * chunko, :, ii : ii + 1, jj : jj + 1],
                    axis=2,
                )
    return out


def _conv2d1(x, w, stride, pad, groups=1):
    """Convolution (equivalent pytorch.conv2d)"""
    if pad != 0:
        x = cp.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), "constant")

    b, ci, hi, wi = x.shape
    co, _, hk, wk = w.shape
    ho = int(cp.floor(1 + (hi - hk) / stride[0]))
    wo = int(cp.floor(1 + (wi - wk) / stride[1]))
    out = cp.zeros([b, co, ho, wo], dtype="float32")
    chunk = ci // groups
    chunko = co // groups

    x = cp.ascontiguousarray(x)
    w = cp.ascontiguousarray(w)
    bs = (32, 32, 1)
    gs = (int(cp.ceil(wo / bs[0])), int(cp.ceil(ho / bs[1])), int(cp.ceil(b / bs[2])))
    conv2d_kernel(
        gs,
        bs,
        (
            out,
            x,
            w,
            stride[0],
            stride[1],
            b,
            co,
            ho,
            wo,
            ci,
            hi,
            wi,
            groups,
            chunk,
            chunko,
            hk,
            wk,
        ),
    )
    return out


def _conv_transpose2d(x, w, stride, pad, groups=1):
    """Transposed convolution (equivalent pytorch.conv_transpose2d)"""
    b, co, ho, wo = x.shape
    co, ci, hk, wk = w.shape

    hi = (ho - 1) * stride[0] + hk
    wi = (wo - 1) * stride[1] + wk
    out = cp.zeros([b, ci, hi, wi], dtype="float32")
    chunk = ci // groups
    chunko = co // groups
    for g in range(groups):
        for ii in range(hk):
            for jj in range(wk):
                x_windows = x[:, g * chunko : (g + 1) * chunko]
                out[
                    :,
                    g * chunk : (g + 1) * chunk,
                    ii : ho * stride[0] + ii : stride[0],
                    jj : wo * stride[1] + jj : stride[1],
                ] += (
                    x_windows
                    * w[g * chunko : (g + 1) * chunko, :, ii : ii + 1, jj : jj + 1]
                )
    if pad != 0:
        out = out[:, :, pad[0] : out.shape[2] - pad[0], pad[1] : out.shape[3] - pad[1]]
    return out


def _conv_transpose2d1(x, w, stride, pad, groups=1):
    """Transposed convolution (equivalent pytorch.conv_transpose2d)"""
    b, co, ho, wo = x.shape
    co, ci, hk, wk = w.shape

    hi = (ho - 1) * stride[0] + hk
    wi = (wo - 1) * stride[1] + wk
    out = cp.zeros([b, ci, hi, wi], dtype="float32")
    chunk = ci // groups
    chunko = co // groups

    mbs = 32
    bs = (mbs, mbs, 1)
    gs = (int(cp.ceil(wo / bs[0])), int(cp.ceil(ho / bs[1])), int(cp.ceil(b / bs[2])))
    x = cp.ascontiguousarray(x)
    w = cp.ascontiguousarray(w)
    conv_transpose2d_kernel(
        gs,
        bs,
        (
            out,
            x,
            w,
            stride[0],
            stride[1],
            b,
            co,
            ho,
            wo,
            ci,
            hi,
            wi,
            groups,
            chunk,
            chunko,
            hk,
            wk,
            mbs,
        ),
        shared_mem=(hk + mbs * stride[0]) * (wk + mbs * stride[1]) * 4,
    )
    if pad != 0:
        out = out[:, :, pad[0] : out.shape[2] - pad[0], pad[1] : out.shape[3] - pad[1]]
    return out


def afb1d(x, h0, h1="zero", dim=-1):
    """1D analysis filter bank (along one dimension only) of an image

    Parameters
    ----------
    x (array): 4D input with the last two dimensions the spatial input
    h0 (array): 4D input for the lowpass filter. Should have shape (1, 1,
        h, 1) or (1, 1, 1, w)
    h1 (array): 4D input for the highpass filter. Should have shape (1, 1,
        h, 1) or (1, 1, 1, w)
    dim (int) - dimension of filtering. d=2 is for a vertical filter (called
        column filtering but filters across the rows). d=3 is for a
        horizontal filter, (called row filtering but filters across the
        columns).

    Returns
    -------
    lohi: lowpass and highpass subbands concatenated along the channel
        dimension
    """
    C = x.shape[1]
    # Convert the dim to positive
    d = dim % 4
    s = (2, 1) if d == 2 else (1, 2)
    N = x.shape[d]
    L = h0.size
    _L2 = L // 2
    shape = [1, 1, 1, 1]
    shape[d] = L
    h = cp.concatenate([h0.reshape(*shape), h1.reshape(*shape)] * C, axis=0)
    # Calculate the pad size
    outsize = pywt.dwt_coeff_len(N, L, mode="symmetric")
    p = 2 * (outsize - 1) - N + L
    pad = (0, 0, p // 2, (p + 1) // 2) if d == 2 else (p // 2, (p + 1) // 2, 0, 0)
    x = _mypad(x, pad=pad)
    # lohi1 = _conv2d(x, h, stride=s, pad=0, groups=C)
    lohi = _conv2d1(x, h, stride=s, pad=0, groups=C)

    # print(cp.linalg.norm(lohi-lohi1))
    return lohi


def sfb1d(lo, hi, g0, g1="zero", dim=-1):
    """1D synthesis filter bank of an image Array"""

    C = lo.shape[1]
    d = dim % 4
    L = g0.size
    shape = [1, 1, 1, 1]
    shape[d] = L
    _N = 2 * lo.shape[d]
    s = (2, 1) if d == 2 else (1, 2)
    g0 = cp.concatenate([g0.reshape(*shape)] * C, axis=0)
    g1 = cp.concatenate([g1.reshape(*shape)] * C, axis=0)
    pad = (L - 2, 0) if d == 2 else (0, L - 2)
    # y1 = _conv_transpose2d(cp.asarray(lo), cp.asarray(g0), stride=s, pad=pad, groups=C) + \
    #     _conv_transpose2d(cp.asarray(hi), cp.asarray(g1), stride=s, pad=pad, groups=C)

    y = _conv_transpose2d1(
        cp.asarray(lo), cp.asarray(g0), stride=s, pad=pad, groups=C
    ) + _conv_transpose2d1(cp.asarray(hi), cp.asarray(g1), stride=s, pad=pad, groups=C)

    # print(cp.linalg.norm(y-y1),cp.linalg.norm(y),cp.linalg.norm(y1))
    return y


class DWTForward:
    """Performs a 2d DWT Forward decomposition of an image

    Args:
        wave (str): Which wavelet to use.
    """

    def __init__(self, wave="db1"):
        super().__init__()

        wave = pywt.Wavelet(wave)
        h0_col, h1_col = wave.dec_lo, wave.dec_hi
        h0_row, h1_row = h0_col, h1_col

        self.h0_col = cp.array(h0_col).astype("float32")[::-1].reshape((1, 1, -1, 1))
        self.h1_col = cp.array(h1_col).astype("float32")[::-1].reshape((1, 1, -1, 1))
        self.h0_row = cp.array(h0_row).astype("float32")[::-1].reshape((1, 1, 1, -1))
        self.h1_row = cp.array(h1_row).astype("float32")[::-1].reshape((1, 1, 1, -1))

    def apply(self, x):
        """Forward pass of the DWT.

        Args:
            x (array): Input of shape :math:`(N, C_{in}, H_{in}, W_{in})`

        Returns:
            (yl, yh)
                tuple of lowpass (yl) and bandpass (yh) coefficients.
                yh is a list of scale coefficients. yl has shape
                :math:`(N, C_{in}, H_{in}', W_{in}')` and yh has shape
                :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. The new
                dimension in yh iterates over the LH, HL and HH coefficients.

        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.
        """
        # Do a multilevel transform
        # Do 1 level of the transform
        lohi = afb1d(x, self.h0_row, self.h1_row, dim=3)
        y = afb1d(lohi, self.h0_col, self.h1_col, dim=2)
        s = y.shape
        y = y.reshape(s[0], -1, 4, s[-2], s[-1])
        x = cp.ascontiguousarray(y[:, :, 0])
        yh = cp.ascontiguousarray(y[:, :, 1:])
        return x, yh


class DWTInverse:
    """Performs a 2d DWT Inverse reconstruction of an image

    Args:
        wave (str): Which wavelet to use.
    """

    def __init__(self, wave="db1"):
        super().__init__()
        wave = pywt.Wavelet(wave)
        g0_col, g1_col = wave.rec_lo, wave.rec_hi
        g0_row, g1_row = g0_col, g1_col
        # Prepare the filters
        self.g0_col = cp.array(g0_col).astype("float32").reshape((1, 1, -1, 1))
        self.g1_col = cp.array(g1_col).astype("float32").reshape((1, 1, -1, 1))
        self.g0_row = cp.array(g0_row).astype("float32").reshape((1, 1, 1, -1))
        self.g1_row = cp.array(g1_row).astype("float32").reshape((1, 1, 1, -1))

    def apply(self, coeffs):
        """
        Args:
            coeffs (yl, yh): tuple of lowpass and bandpass coefficients, where:
              yl is a lowpass array of shape :math:`(N, C_{in}, H_{in}',
              W_{in}')` and yh is a list of bandpass arrays of shape
              :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. I.e. should match
              the format returned by DWTForward

        Returns:
            Reconstructed input of shape :math:`(N, C_{in}, H_{in}, W_{in})`

        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.

        """
        yl, yh = coeffs
        lh = yh[:, :, 0]
        hl = yh[:, :, 1]
        hh = yh[:, :, 2]
        lo = sfb1d(yl, lh, self.g0_col, self.g1_col, dim=2)
        hi = sfb1d(hl, hh, self.g0_col, self.g1_col, dim=2)
        yl = sfb1d(lo, hi, self.g0_row, self.g1_row, dim=3)
        return yl


def remove_stripe_fw(data, sigma, wname, level):
    """Remove stripes with wavelet filtering"""

    [nproj, nz, ni] = data.shape

    nproj_pad = nproj + nproj // 8
    _xshift = int((nproj_pad - nproj) // 2)

    # Accepts all wave types available to PyWavelets
    xfm = DWTForward(wave=wname)
    ifm = DWTInverse(wave=wname)

    # Wavelet decomposition.
    cc = []
    sli = cp.zeros([nz, 1, nproj_pad, ni], dtype="float32")

    sli[:, 0, (nproj_pad - nproj) // 2 : (nproj_pad + nproj) // 2] = data.astype(
        "float32"
    ).swapaxes(0, 1)
    for k in range(level):
        sli, c = xfm.apply(sli)
        cc.append(c)
        # FFT
        fcV = cp.fft.fft(cc[k][:, 0, 1], axis=1)
        _, my, mx = fcV.shape
        # Damping of ring artifact information.
        y_hat = cp.fft.ifftshift((cp.arange(-my, my, 2) + 1) / 2)
        damp = -cp.expm1(-(y_hat**2) / (2 * sigma**2))
        fcV *= cp.tile(damp, (mx, 1)).swapaxes(0, 1)
        # Inverse FFT.
        cc[k][:, 0, 1] = cp.fft.ifft(fcV, my, axis=1).real

    # # Wavelet reconstruction.
    for k in range(level)[::-1]:
        shape0 = cc[k][0, 0, 1].shape
        sli = sli[:, :, : shape0[0], : shape0[1]]
        sli = ifm.apply((sli, cc[k]))

    data = sli[:, 0, (nproj_pad - nproj) // 2 : (nproj_pad + nproj) // 2, :ni].astype(
        data.dtype
    )  # modified
    data = data.swapaxes(0, 1)

    return data


######## Titarenko ring removal ################
def remove_stripe_ti(data, beta, mask_size):
    """Remove stripes with a new method by V. Titareno"""
    gamma = beta * ((1 - beta) / (1 + beta)) ** cp.abs(
        cp.fft.fftfreq(data.shape[-1]) * data.shape[-1]
    )
    gamma[0] -= 1
    v = cp.mean(data, axis=0)
    v = v - v[:, 0:1]
    v = cp.fft.irfft(cp.fft.rfft(v) * cp.fft.rfft(gamma))
    mask = cp.zeros(v.shape, dtype=v.dtype)
    mask_size = mask_size * mask.shape[1]
    mask[
        :, mask.shape[1] // 2 - mask_size // 2 : mask.shape[1] // 2 + mask_size // 2
    ] = 1
    data[:] += v * mask
    return data


######## Optimized version for Vo-all ring removal in tomopy#########
def _rs_sort(sinogram, size, matindex, dim):
    """
    Remove stripes using the sorting technique.
    """
    sinogram = cp.transpose(sinogram)
    matcomb = cp.asarray(cp.dstack((matindex, sinogram)))

    # matsort = cp.asarray([row[row[:, 1].argsort()] for row in matcomb])
    ids = cp.argsort(matcomb[:, :, 1], axis=1)
    matsort = matcomb.copy()
    matsort[:, :, 0] = cp.take_along_axis(matsort[:, :, 0], ids, axis=1)
    matsort[:, :, 1] = cp.take_along_axis(matsort[:, :, 1], ids, axis=1)
    if dim == 1:
        matsort[:, :, 1] = median_filter(matsort[:, :, 1], (size, 1))
    else:
        matsort[:, :, 1] = median_filter(matsort[:, :, 1], (size, size))

    # matsortback = cp.asarray([row[row[:, 0].argsort()] for row in matsort])

    ids = cp.argsort(matsort[:, :, 0], axis=1)
    matsortback = matsort.copy()
    matsortback[:, :, 0] = cp.take_along_axis(matsortback[:, :, 0], ids, axis=1)
    matsortback[:, :, 1] = cp.take_along_axis(matsortback[:, :, 1], ids, axis=1)

    sino_corrected = matsortback[:, :, 1]
    return cp.transpose(sino_corrected)


def _mpolyfit(x, y):
    n = len(x)
    x_mean = cp.mean(x)
    y_mean = cp.mean(y)

    Sxy = cp.sum(x * y) - n * x_mean * y_mean
    Sxx = cp.sum(x * x) - n * x_mean * x_mean

    slope = Sxy / Sxx
    intercept = y_mean - slope * x_mean
    return slope, intercept


def _detect_stripe(listdata, snr):
    """
    Algorithm 4 in :cite:`Vo:18`. Used to locate stripes.
    """
    numdata = len(listdata)
    listsorted = cp.sort(listdata)[::-1]
    xlist = cp.arange(0, numdata, 1.0)
    ndrop = cp.int16(0.25 * numdata)
    # (_slope, _intercept) = cp.polyfit(xlist[ndrop:-ndrop - 1],
    #   listsorted[ndrop:-ndrop - 1], 1)
    (_slope, _intercept) = _mpolyfit(
        xlist[ndrop : -ndrop - 1], listsorted[ndrop : -ndrop - 1]
    )

    numt1 = _intercept + _slope * xlist[-1]
    noiselevel = cp.abs(numt1 - _intercept)
    noiselevel = cp.clip(noiselevel, 1e-6, None)
    val1 = cp.abs(listsorted[0] - _intercept) / noiselevel
    val2 = cp.abs(listsorted[-1] - numt1) / noiselevel
    listmask = cp.zeros_like(listdata)
    if val1 >= snr:
        upper_thresh = _intercept + noiselevel * snr * 0.5
        listmask[listdata > upper_thresh] = 1.0
    if val2 >= snr:
        lower_thresh = numt1 - noiselevel * snr * 0.5
        listmask[listdata <= lower_thresh] = 1.0
    return listmask


def _rs_large(sinogram, snr, size, matindex, drop_ratio=0.1, norm=True):
    """
    Remove large stripes.
    """
    drop_ratio = max(min(drop_ratio, 0.8), 0)  # = cp.clip(drop_ratio, 0.0, 0.8)
    (nrow, ncol) = sinogram.shape
    ndrop = int(0.5 * drop_ratio * nrow)
    sinosort = cp.sort(sinogram, axis=0)
    sinosmooth = median_filter(sinosort, (1, size))
    list1 = cp.mean(sinosort[ndrop : nrow - ndrop], axis=0)
    list2 = cp.mean(sinosmooth[ndrop : nrow - ndrop], axis=0)
    # listfact = cp.divide(list1,
    #                      list2,
    #                      out=cp.ones_like(list1),
    #                      where=list2 != 0)

    listfact = list1 / list2

    # Locate stripes
    listmask = _detect_stripe(listfact, snr)
    listmask = binary_dilation(listmask, iterations=1).astype(listmask.dtype)
    matfact = cp.tile(listfact, (nrow, 1))
    # Normalize
    if norm is True:
        sinogram = sinogram / matfact
    sinogram1 = cp.transpose(sinogram)
    matcombine = cp.asarray(cp.dstack((matindex, sinogram1)))

    # matsort = cp.asarray([row[row[:, 1].argsort()] for row in matcombine])
    ids = cp.argsort(matcombine[:, :, 1], axis=1)
    matsort = matcombine.copy()
    matsort[:, :, 0] = cp.take_along_axis(matsort[:, :, 0], ids, axis=1)
    matsort[:, :, 1] = cp.take_along_axis(matsort[:, :, 1], ids, axis=1)

    matsort[:, :, 1] = cp.transpose(sinosmooth)
    # matsortback = cp.asarray([row[row[:, 0].argsort()] for row in matsort])
    ids = cp.argsort(matsort[:, :, 0], axis=1)
    matsortback = matsort.copy()
    matsortback[:, :, 0] = cp.take_along_axis(matsortback[:, :, 0], ids, axis=1)
    matsortback[:, :, 1] = cp.take_along_axis(matsortback[:, :, 1], ids, axis=1)

    sino_corrected = cp.transpose(matsortback[:, :, 1])
    listxmiss = cp.where(listmask > 0.0)[0]
    sinogram[:, listxmiss] = sino_corrected[:, listxmiss]
    return sinogram


def _rs_dead(sinogram, snr, size, matindex, norm=True):
    """
    Remove unresponsive and fluctuating stripes.
    """
    sinogram = cp.copy(sinogram)  # Make it mutable
    (nrow, _) = sinogram.shape
    # sinosmooth = cp.apply_along_axis(uniform_filter1d, 0, sinogram, 10)
    sinosmooth = uniform_filter1d(sinogram, 10, axis=0)

    listdiff = cp.sum(cp.abs(sinogram - sinosmooth), axis=0)
    listdiffbck = median_filter(listdiff, size)

    listfact = listdiff / listdiffbck

    listmask = _detect_stripe(listfact, snr)
    listmask = binary_dilation(listmask, iterations=1).astype(listmask.dtype)
    listmask[0:2] = 0.0
    listmask[-2:] = 0.0
    listx = cp.where(listmask < 1.0)[0]
    _listy = cp.arange(nrow)
    matz = sinogram[:, listx]

    listxmiss = cp.where(listmask > 0.0)[0]

    # finter = interpolate.interp2d(listx.get(), listy.get(), matz.get(), kind='linear')
    if len(listxmiss) > 0:
        # sinogram_c[:, listxmiss.get()] = finter(listxmiss.get(), listy.get())
        ids = cp.searchsorted(listx, listxmiss)
        sinogram[:, listxmiss] = matz[:, ids - 1] + (listxmiss - listx[ids - 1]) * (
            matz[:, ids] - matz[:, ids - 1]
        ) / (listx[ids] - listx[ids - 1])

    # Remove residual stripes
    if norm is True:
        sinogram = _rs_large(sinogram, snr, size, matindex)
    return sinogram


def _create_matindex(nrow, ncol):
    """
    Create a 2D array of indexes used for the sorting technique.
    """
    listindex = cp.arange(0.0, ncol, 1.0)
    matindex = cp.tile(listindex, (nrow, 1))
    return matindex


def remove_all_stripe(tomo, snr=3, la_size=61, sm_size=21, dim=1):
    """
    Remove all types of stripe artifacts from sinogram using Nghia Vo's
    approach :cite:`Vo:18` (combination of algorithm 3,4,5, and 6).

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    snr  : float
        Ratio used to locate large stripes.
        Greater is less sensitive.
    la_size : int
        Window size of the median filter to remove large stripes.
    sm_size : int
        Window size of the median filter to remove small-to-medium stripes.
    dim : {1, 2}, optional
        Dimension of the window.

    Returns
    -------
    ndarray
        Corrected 3D tomographic data.
    """
    matindex = _create_matindex(tomo.shape[2], tomo.shape[0])
    for m in range(tomo.shape[1]):
        sino = tomo[:, m, :]
        sino = _rs_dead(sino, snr, la_size, matindex)
        sino = _rs_sort(sino, sm_size, matindex, dim)
        tomo[:, m, :] = sino
    return tomo
