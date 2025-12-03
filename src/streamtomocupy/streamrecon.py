import cupy as cp
import numpy as np
import sys

from streamtomocupy import rec
from streamtomocupy import proc
from streamtomocupy.chunking import gpu_batch

cp.cuda.set_pinned_memory_allocator(cp.cuda.PinnedMemoryPool().malloc)


class StreamRecon:
    """Streaming reconstruction"""

    def __init__(self, args):
        ni = args.n
        nproj = args.nproj
        nz = args.nz
        nflat = args.nflat
        ndark = args.ndark
        in_dtype = args.in_dtype
        ngpus = args.ngpus
        
        if ni%2 != 0:
            raise Exception('data size in the last dimension should be even')

        if (args.file_type == 'double_fov'):
            n = 2*ni
        else:
            n = ni

        # chunk sizes
        ncz = args.nsino_per_chunk
        ncproj = args.nproj_per_chunk

        # allocate gpu and pinned memory buffers
        # calculate max buffer size
        nbytes1 = (
            2
            * (nproj * ncz * ni + nflat * ncz * ni + ndark * ncz * ni)
            * np.dtype(in_dtype).itemsize
        )
        nbytes1 += 2 * (nproj * ncz * ni) * np.dtype(args.dtype).itemsize

        nbytes2 = 2 * (ncproj * nz * ni) * np.dtype(args.dtype).itemsize
        nbytes2 += 2 * (ncproj * nz * ni) * np.dtype(args.dtype).itemsize

        nbytes3 = 2 * (nproj * ncz * ni) * np.dtype(args.dtype).itemsize
        nbytes3 += 2 * (ncz * n * n) * np.dtype(args.dtype).itemsize

        nbytes4 = (
            2
            * (nproj * ncz * ni + nflat * ncz * ni + ndark * ncz * ni)
            * np.dtype(in_dtype).itemsize
        )
        nbytes4 += 2 * (ncz * n * n) * np.dtype(args.dtype).itemsize
        nbytes = max(nbytes1, nbytes2, nbytes3, nbytes4)

        # create CUDA streams and allocate pinned memory
        self.stream = [[[] for _ in range(3)] for _ in range(ngpus)]
        self.pinned_mem = [[] for _ in range(ngpus)]
        self.gpu_mem = [[] for _ in range(ngpus)]

        for igpu in range(ngpus):
            with cp.cuda.Device(igpu):
                self.pinned_mem[igpu] = cp.cuda.alloc_pinned_memory(nbytes)
                self.gpu_mem[igpu] = cp.cuda.alloc(nbytes)
                for k in range(3):
                    self.stream[igpu][k] = cp.cuda.Stream(non_blocking=False)

        # classes for processing
        self.cl_rec = rec.Rec(args, nproj, ncz, n, ni, ngpus)
        self.cl_proc = proc.Proc(args)

        # intermediate arrays with results
        self.res = [None] * 3
        self.res[0] = np.empty([nproj, nz, ni], dtype=args.dtype)
        self.res[1] = np.empty([nproj, nz, ni], dtype=args.dtype)
        self.res[2] = np.empty([nz, n, n], dtype=args.dtype)

        # gpu batch parameters
        self.ncz = ncz
        self.ncproj = ncproj
        self.ngpus = ngpus
        self.ni = ni
        self.args = args

        print("class created")

    def proc_sino(self, res, data, dark, flat):
        @gpu_batch(self.ncz, self.ngpus, axis_out=1, axis_inp=1)
        def _proc_sino(self, res, data, dark, flat):
            """Processing a sinogram data chunk"""

            self.cl_proc.remove_outliers(data)
            self.cl_proc.remove_outliers(dark)
            self.cl_proc.remove_outliers(flat)
            res[:] = self.cl_proc.darkflat_correction(data, dark, flat)
            self.cl_proc.remove_stripe(res)

        return _proc_sino(self, res, data, dark, flat)

    def proc_proj(self, res, data):
        @gpu_batch(self.ncproj, self.ngpus, axis_out=0, axis_inp=0)
        def _proc_proj(self, res, data):
            """Processing a projection data chunk"""

            self.cl_proc.retrieve_phase(data)
            self.cl_proc.minus_log(data)
            res[:] = data[:]

        return _proc_proj(self, res, data)

    def rec_sino(self, res, data, theta):
        @gpu_batch(self.ncz, self.ngpus, axis_out=0, axis_inp=1)
        def _rec_sino(self, res, data, theta):
            """Filtered backprojection with sinogram data chunks"""
            data = cp.ascontiguousarray(data.swapaxes(0, 1))
            data = self.cl_rec.pad360(data)  # may change data shape
            self.cl_rec.fbp_filter_center(data)
            self.cl_rec.backprojection(res, data, theta)

        return _rec_sino(self, res, data, theta)

    def rec(self, data, dark, flat, theta):
        @gpu_batch(self.ncz, self.ngpus, axis_out=0, axis_inp=1)
        def _rec(self, res, data, dark, flat, theta):
            """Processing + filtered backprojection with sinogram data chunks"""

            self.cl_proc.remove_outliers(data)
            self.cl_proc.remove_outliers(dark)
            self.cl_proc.remove_outliers(flat)
            data = self.cl_proc.darkflat_correction(
                data, dark, flat
            )  # may change data type
            self.cl_proc.remove_stripe(data)
            self.cl_proc.minus_log(data)
            data = cp.ascontiguousarray(data.swapaxes(0, 1))
            data = self.cl_rec.pad360(data)  # may change data shape
            self.cl_rec.fbp_filter_center(data)
            self.cl_rec.backprojection(res, data, theta)

        return _rec(self, self.res[2], data, dark, flat, theta)

    def rec_steps(self, data, dark, flat, theta):
        """Processing with sinogram and projection data chunks,
        filtered backprojection with sinogram data chunks"""
        self.proc_sino(self.res[0], data, dark, flat)
        self.proc_proj(self.res[1], self.res[0])
        self.rec_sino(self.res[2], self.res[1], theta)

    def get_res(self):
        """Get intermediate results for each step"""
        return self.res
