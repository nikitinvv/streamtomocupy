import numpy as np
import cupy as cp
import time
import h5py

from streamtomocupy import config
from streamtomocupy import streamrecon
import tifffile

cp.cuda.set_pinned_memory_allocator(cp.cuda.PinnedMemoryPool().malloc)


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
args = config.read_args("goran.conf")

with h5py.File(
    "/das/work/p13/p13657/sandbox_viktor_goran/data/R108dC03a_S08_001_withAngles.h5",
    "r",
) as fid:
    proj = fid["exchange/data"][:]
    flat = fid["exchange/data_white_pre"][:]
    dark = fid["exchange/data_dark"][:]
    theta = (fid["exchange/theta"][:] / 180 * np.pi).astype(
        "float32"
    )  # note work with float32 angles only
    print(f"{proj.shape=}, {flat.shape=}, {dark.shape=}, {theta[:10]=}")
args = get_data_pars(args, proj, flat, dark)

# streaming reconstruction class
t = time.time()
cl_recstream = streamrecon.StreamRecon(args)
print("Create class, time", time.time() - t)

res = cl_recstream.get_res()

# processing and reconstruction
t = time.time()

# can specify slices to process
st = 0
end = 2160

t = time.time()
cl_recstream.proc_sino(
    res[0][:, st:end], proj[:, st:end], dark[:, st:end], flat[:, st:end]
)

# parameters can be changed between steps, in a loop for instance..
# args.retrieve_phase_alpha = 0.01
cl_recstream.proc_proj(res[1][:, st:end], res[0][:, st:end])

# args.rotation_axis = 70
cl_recstream.rec_sino(res[2][st:end], res[1][:, st:end], theta)

print(
    "Manual processing and reconstruction by sinogram and projection chunks, time",
    time.time() - t,
)

tifffile.imwrite(
    "/das/work/p13/p13657/sandbox_viktor_goran/data/R108dC03a_S08_001_withAngles.tif",
    res[2][(end + st) // 2].astype("float32"),
)
# print('norm of the result', np.linalg.norm(res[2][st:end].astype('float32')))
