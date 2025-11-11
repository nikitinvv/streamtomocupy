import cupy as cp

from streamtomocupy import cfunc_fourierrec
from streamtomocupy import cfunc_fourierrecfp16


class FourierRec:
    """Backprojection by the Fourier-based method"""

    def __init__(self, n, nproj, nz, dtype):
        self.nz = nz
        self.n = n
        self.nproj = nproj
        if dtype == "float16":
            self.fslv = cfunc_fourierrecfp16.cfunc_fourierrec(nproj, nz // 2, n)
        else:
            self.fslv = cfunc_fourierrec.cfunc_fourierrec(nproj, nz // 2, n)

    def backprojection(self, obj, data, theta, stream):
        # reorganize data as a complex array, reuse data
        data = cp.ascontiguousarray(
            cp.concatenate(
                (
                    data[: self.nz // 2, :, :, cp.newaxis],
                    data[self.nz // 2 :, :, :, cp.newaxis],
                ),
                axis=3,
            ).reshape(data.shape)
        )
        # reuse obj array
        objc = cp.ascontiguousarray(obj.reshape(self.nz // 2, self.n, 2 * self.n))
        self.fslv.backprojection(
            obj.data.ptr, data.data.ptr, theta.data.ptr, len(theta), stream.ptr
        )
        obj[:] = cp.concatenate((objc[:, :, ::2], objc[:, :, 1::2]))
