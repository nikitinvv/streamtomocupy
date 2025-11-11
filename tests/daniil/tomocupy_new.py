#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 10:08:38 2024

@author: DLS Imaging Team
"""

# All external imports
import numpy as np
import logging
import time
import argparse
import os
import timeit
import h5py


# TomoCuPy imports
import streamtomocupy.config as sconfig
from streamtomocupy import streamrecon
from streamtomocupy import find_center_vo
import tifffile


from logging.config import dictConfig

config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": """%(levelname)-10s: %(name)-10s: %(funcName)20s() line %(lineno)d: %(message)s"""
        }
    },
    "handlers": {
        "default": {
            "level": "DEBUG",
            "formatter": "standard",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": "logs/tomocupy.log",
        },
    },
    "loggers": {
        "": {
            "level": "WARNING",
            "handlers": ["default", "file"],
            "class": "logging.FileHandler",
            "filename": "logs/tomocupy.log",
            "propagate": True,
        },
        "my_logger": {
            "level": "DEBUG",
            "handlers": ["default", "file"],
            "propagate": False,
        },
    },
}
dictConfig(config)
log = logging.getLogger("my_logger")


def timeit_my(func):
    def gettime(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed = end - start
        log.info(f"Time taken: {elapsed:.6f} seconds")
        return result

    return gettime


@timeit_my
def hdf5_loader(path_to_data):
    log.info("Loading data")
    h5f = h5py.File(path_to_data, "r")

    keys = h5f["entry1/flyScanDetector/image_key"][:]
    projdata = h5f["entry1/flyScanDetector/data"][:]
    angles = np.float32(h5f["entry1/flyScanDetector/zebraSM1"][:])

    log.info("Loading complete")
    h5f.close()
    return (
        projdata[keys == 1, :, :],
        projdata[keys == 2, :, :],
        angles[keys == 0],
        projdata[keys == 0, :, :],
    )


@timeit_my
def CoR_calc(data, darks, flats):
    log.info("CoR estimation")
    center_search_width = 100
    center_search_step = 0.5
    center_search_ind = data.shape[1] // 2
    rotation_axis = find_center_vo(
        data[:, center_search_ind],
        darks[:, center_search_ind],
        flats[:, center_search_ind],
        smin=-center_search_width,
        smax=center_search_width,
        step=center_search_step,
    )
    log.info("center of rotation: {}".format(rotation_axis))
    log.info("CoR estimation complete")
    return rotation_axis


@timeit_my
def init(data_shape, flat_shape, dark_shape, dtype):
    log.info("StreamTomocuPy init")

    args = sconfig.read_args("daniil.conf")
    args.nproj = data_shape[0]
    args.nz = data_shape[1]
    args.n = data_shape[2]
    args.nflat = flat_shape[0]
    args.ndark = dark_shape[0]
    args.in_dtype = dtype
    cl_recstream = streamrecon.StreamRecon(args)
    return cl_recstream


@timeit_my
def reconstruction_streamtomocupy(cl, data, darks, flats, rotation_axis, angles):
    log.info("StreamTomocuPy Reconstruction")
    cl.args.rotation_axis = rotation_axis
    # VN: note 'fw' filter makes it slower, you can check 'none' for comparison
    # cl.args.remove_stripe_method = 'none'
    cl.rec(data, darks, flats, angles)
    res = cl.get_res()[2]
    log.info("Reconstruction by sinogram chunks is complete")
    return res


@timeit_my
def reconstruction_streamtomocupy_steps(cl, data, darks, flats, rotation_axis, angles):
    log.info("StreamTomocuPy Reconstruction in 3 steps")
    cl.args.rotation_axis = rotation_axis
    cl.args.retrieve_phase_method = "paganin"

    # VN: maybe more adjustments for paganin
    cl.args.fbp_filter = "shepp"
    cl.args.retrieve_phase_alpha = 0.003

    # VN: note 'fw' filter makes it slower, you can check 'none' for comparison
    # cl.args.remove_stripe_method = 'none'

    cl.rec_steps(data, darks, flats, angles)
    res = cl.get_res()[2]
    log.info(
        "Reconstruction by sinogram and projections chunks (with steps) is complete"
    )
    return res


@timeit_my
def save_images(data, path_to_folder):
    log.info("Perform saving of data into 3D tiff")
    tifffile.imwrite(path_to_folder + "reconstruction_tomocupy.tiff", data)
    log.info("Saving complete")
    return True


def run_streamtomocupy_pipeline(path_to_data: str, output_folder: str) -> int:
    start_time = timeit.default_timer()

    # StreamTomoCuPy's Pipeline
    flats, darks, angles, data = hdf5_loader(path_to_data)

    # formating for tomocupy input
    angles = (angles / 180 * np.pi).astype("float32")

    cl = init(data.shape, flats.shape, darks.shape, data.dtype)
    rotation_axis = CoR_calc(data, darks, flats)
    reconstruction = reconstruction_streamtomocupy(
        cl, data, darks, flats, rotation_axis, angles
    )
    # VN: if phase retrieval is needed run this instead
    # reconstruction = reconstruction_streamtomocupy_steps(cl, data, darks, flats, rotation_axis, angles)
    data_saved = save_images(reconstruction, output_folder)

    txtstr = "%s = %.3fs" % (
        "Total elapsed time for the pipeline",
        timeit.default_timer() - start_time,
    )
    log.info(txtstr)
    return data_saved


def get_args():
    parser = argparse.ArgumentParser(
        description="Script that executes a TomoCuPy's pipeline."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        help="A path to the data.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="./",
        help="Directory to save the results (images).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    current_dir = os.path.basename(os.path.abspath(os.curdir))
    args = get_args()
    path_to_data = args.input
    output_folder = args.output
    return_val = run_streamtomocupy_pipeline(path_to_data, output_folder)
    if return_val:
        print("The tomocupy pipeline successfully completed")
