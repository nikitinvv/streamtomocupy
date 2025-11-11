import numpy as np
import time
import h5py

from streamtomocupy import config
from streamtomocupy import streamrecon


def get_data_pars(args, proj, flat, dark):
    """Get parameters of the data"""

    args.nproj = proj.shape[0]
    args.nz = proj.shape[1]
    args.n = proj.shape[2]
    args.nflat = flat.shape[0]
    args.ndark = dark.shape[0]
    args.in_dtype = proj.dtype
    return args


# init parameters with default values. can be done ones
# config.write_args('test.conf')
# read parameters
args = config.read_args("test.conf")

with h5py.File("test_data.h5", "r") as fid:
    proj = fid["exchange/data"][:]
    flat = fid["exchange/data_white"][:]
    dark = fid["exchange/data_dark"][:]
    theta = fid["exchange/theta"][:] / 180 * np.pi
args = get_data_pars(args, proj, flat, dark)

# streaming reconstruction class
t = time.time()
cl_recstream = streamrecon.StreamRecon(args)
print("Create class, time", time.time() - t)

res = cl_recstream.get_res()

# processing and reconstruction
t = time.time()
cl_recstream.rec_steps(proj, dark, flat, theta)
print("Reconstruction by sinogram and projection chunks, time", time.time() - t)
print("norm of the result", np.linalg.norm(res[2].astype("float32")))
