import cupy as cp

from streamtomocupy import fourierrec
from streamtomocupy import lprec
from streamtomocupy import linerec
from streamtomocupy import fbp_filter


class Rec:
    def __init__(self, args, nproj, ncz, n, ni, ngpus):
        self.cl_rec = [None] * ngpus
        self.cl_filter = [None] * ngpus
        self.wfilter = [None] * ngpus
        self.theta = [None] * ngpus
        ne = 4 * n  # filter oversampling
        for igpu in range(ngpus):
            with cp.cuda.Device(igpu):
                if args.reconstruction_algorithm == "fourierrec":
                    self.cl_rec[igpu] = fourierrec.FourierRec(n, nproj, ncz, args.dtype)
                elif args.reconstruction_algorithm == "lprec":
                    self.cl_rec[igpu] = lprec.LpRec(n, nproj, ncz, args.dtype)
                elif args.reconstruction_algorithm == "linerec":
                    self.cl_rec[igpu] = linerec.LineRec(
                        nproj, nproj, 22, ncz, n, args.dtype
                    )

                self.cl_filter[igpu] = fbp_filter.FBPFilter(ne, nproj, ncz, args.dtype)
                # calculate the FBP filter with quadrature rules
        self.wfilter[0] = self.cl_filter[0].calc_filter(args.fbp_filter)
        for igpu in range(ngpus):
            with cp.cuda.Device(igpu):
                self.wfilter[igpu] = cp.asarray(self.wfilter[0])

        self.ne = ne
        self.n = n
        self.ni = ni
        self.args = args

    def backprojection(self, res, data, theta):
        """Backprojection"""
        igpu = data.device.id
        stream = cp.cuda.get_current_stream()
        self.cl_rec[igpu].backprojection(res, data, theta, stream)

    def center_fix(self):
        """Adjust rotation axis since it becomes different when pad360 is done, and for lrpec"""
        if self.args.rotation_axis == -1:
            center = self.ni // 2
        else:
            center = self.args.rotation_axis
        centeri = center
        if (
            self.args.file_type == "double_fov" and center < self.ni // 2
        ):  # if rotation center is on the left side of the ROI
            center = self.ni - center
        if self.args.reconstruction_algorithm == "lprec":
            center += 0.5
            centeri += 0.5
        return center, centeri

    def pad360(self, data):
        """Pad data with 0 to handle 360 degrees scan"""
        args = self.args
        if args.file_type == "double_fov":
            center, centeri = self.center_fix()
            if centeri < self.ni // 2:
                # if rotation center is on the left side of the ROI
                data[:] = data[:, :, ::-1]
            w = max(1, int(2 * (self.ni - center)))
            # smooth transition at the border
            v = cp.linspace(1, 0, w, endpoint=False)
            v = v**5 * (126 - 420 * v + 540 * v**2 - 315 * v**3 + 70 * v**4)
            data[:, :, -w:] *= v
            # double sinogram size with adding 0
            data = cp.pad(data, ((0, 0), (0, 0), (0, data.shape[-1])), "constant")
        return data

    def fbp_filter_center(self, data):
        """FBP filtering of projections with applying the rotation center shift wrt to the origin"""
        igpu = data.device.id
        stream = cp.cuda.get_current_stream()
        tmp = cp.pad(
            data,
            ((0, 0), (0, 0), (self.ne // 2 - self.n // 2, self.ne // 2 - self.n // 2)),
            mode="edge",
        )
        t = cp.fft.rfftfreq(self.ne).astype("float32")
        center, _ = self.center_fix()
        shift = cp.tile(cp.float32(-center + self.n / 2), [data.shape[0], 1, 1])
        w = self.wfilter[igpu] * cp.exp(-2 * cp.pi * 1j * t * shift)
        self.cl_filter[igpu].filter(tmp, w, stream)
        data[:] = tmp[:, :, self.ne // 2 - self.n // 2 : self.ne // 2 + self.n // 2]

        return data  # reuse input memory
