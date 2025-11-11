import cupy as cp
import numpy as np
import cupyx.scipy.ndimage as ndimage

from streamtomocupy import remove_stripe
from streamtomocupy import retrieve_phase
from streamtomocupy.utils import place_kernel  # tmp


class Proc:
    def __init__(self, args):
        self.args = args

    def darkflat_correction(self, data, dark, flat):
        """Dark-flat field correction"""
        args = self.args
        dark0 = dark.astype(args.dtype, copy=False)
        flat0 = flat.astype(args.dtype, copy=False)
        # works only for processing all angles
        flat0 = cp.mean(flat0, axis=0)
        dark0 = cp.mean(dark0, axis=0)
        res = (data.astype(args.dtype, copy=False) - dark0) / (flat0 - dark0 + 1e-5)
        return res

    def remove_outliers(self, data):
        """Remove outliers"""
        args = self.args

        if int(args.dezinger) > 0:
            w = int(args.dezinger)
            if len(data.shape) == 3:
                fdata = ndimage.median_filter(data, [w, 1, w])
            else:
                fdata = ndimage.median_filter(data, [w, w])
            data[:] = cp.where(
                cp.logical_and(data > fdata, (data - fdata) > args.dezinger_threshold),
                fdata,
                data,
            )
        return data

    def minus_log(self, data):
        """Taking negative logarithm"""
        args = self.args

        if args.minus_log == "True":
            # the following python code makes device synchronization,
            # see warning in https://docs.cupy.dev/en/stable/reference/generated/cupy.place.html
            # and https://github.com/cupy/cupy/blob/118ade4a146d1cc68519f7f661f2c145f0b942c9/cupy/_indexing/insert.py#L35

            # data[data<=0] = 1
            # data[:] = -cp.log(data)
            # data[cp.isnan(data)] = 6.0
            # data[cp.isinf(data)] = 0

            # we temporarily replace it with the code which is not synchrotnized, see place_kernel in utils
            bs = (32, 32, 1)
            gs = (
                int(np.ceil(data.shape[2] / bs[0])),
                int(np.ceil(data.shape[1] / bs[1])),
                int(np.ceil(data.shape[0] / bs[2])),
            )
            data_tmp = cp.ascontiguousarray(data.astype("float32"))
            place_kernel(
                gs, bs, (data_tmp, data.shape[2], data.shape[1], data.shape[0])
            )
            data[:] = data_tmp.astype(args.dtype)
        return data

    def remove_stripe(self, res):
        """Remove stripes"""
        args = self.args

        if args.remove_stripe_method == "fw":
            res[:] = remove_stripe.remove_stripe_fw(
                res, args.fw_sigma, args.fw_filter, args.fw_level
            )
        elif args.remove_stripe_method == "ti":
            res[:] = remove_stripe.remove_stripe_ti(res, args.ti_beta, args.ti_mask)
        elif args.remove_stripe_method == "vo-all":
            res[:] = remove_stripe.remove_all_stripe(
                res,
                args.vo_all_snr,
                args.vo_all_la_size,
                args.vo_all_sm_size,
                args.vo_all_dim,
            )
        return res

    def retrieve_phase(self, data):
        """Retrieve phase"""
        args = self.args

        if (
            args.retrieve_phase_method == "Gpaganin"
            or args.retrieve_phase_method == "paganin"
        ):
            data[:] = retrieve_phase.paganin_filter(
                data,
                args.pixel_size * 1e-4,
                args.propagation_distance / 10,
                args.energy,
                args.retrieve_phase_alpha,
                args.retrieve_phase_method,
                args.retrieve_phase_delta_beta,
                args.retrieve_phase_W * 1e-4,
            )  # units adjusted based on the tomopy implementation
        return data
