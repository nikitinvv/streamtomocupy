import cupy as cp

from streamtomocupy import cfunc_linerec
from streamtomocupy import cfunc_linerecfp16


class LineRec:
    """Backprojection by summation over lines"""

    def __init__(self, nproj, ncproj, nz, ncz, n, dtype):
        self.nproj = nproj
        self.ncproj = ncproj
        self.nz = nz
        self.ncz = ncz
        self.n = n
        self.dtype = dtype
        if dtype == "float16":
            self.fslv = cfunc_linerecfp16.cfunc_linerec(nproj, nz, n, ncproj, ncz)
        else:
            self.fslv = cfunc_linerec.cfunc_linerec(nproj, nz, n, ncproj, ncz)

    def backprojection(self, f, data, theta=[], stream=0, lamino_angle=0, sz=0):
        # may be used for lamino in future..
        phi = cp.pi / 2 + (lamino_angle) / 180 * cp.pi
        self.fslv.backprojection(
            f.data.ptr, data.data.ptr, theta.data.ptr, phi, sz, stream.ptr
        )
