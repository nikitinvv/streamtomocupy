# nsys -profile
import numpy as np
import time

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


args = config.read_args("test.conf")
proj = 100 * np.ones([2048, 2048, 2048], dtype="uint16")
dark = np.zeros([20, 2048, 2048], dtype="uint16")
flat = 200 * np.ones([10, 2048, 2048], dtype="uint16")
theta = np.linspace(0, 2 * np.pi, 2048).astype("float32")

args = get_data_pars(args, proj, flat, dark)

args.reconstruction_algorithm = "lprec"
args.dtype = "float16"
args.nsino_per_chunk = 16
args.nproj_per_chunk = 16
# streaming reconstruction class
t = time.time()
cl_recstream = streamrecon.StreamRecon(args)
print("Class creation time:", time.time() - t)

# processing and reconstruction
t = time.time()
cl_recstream.rec(proj, dark, flat, theta)
print("Reconstruction time:", time.time() - t)
